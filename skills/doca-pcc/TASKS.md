# DOCA PCC workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop
(single small-algorithm smoke → traffic-affecting smoke →
sustained run → loop back if a precondition changes), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the PCC capability surface, the
two-side-program model, the `doca_pcc` per-instance context,
the loaded PCC algorithm image, the attach-to-port semantics,
the triple-axis capability-discovery rule, the env-precondition
matrix, the error taxonomy, the observability surface, and the
safety policy, see [CAPABILITIES.md](CAPABILITIES.md). For the
cross-library DOCA patterns layered under everything below
(the universal Core lifecycle, the cross-library
`DOCA_ERROR_*` taxonomy, the modify-a-shipped-sample
workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a `doca_pcc` context against a BlueField port
that carries the user's RDMA / RoCE traffic, load the user's
custom PCC algorithm image (compiled by `dpacc` from their
DPA-side algorithm source) onto it, parameterize the algorithm,
and confirm both the host side and the BlueField (DPA +
firmware) are in a state where starting the context will
attach the algorithm to live traffic.

Steps the agent should walk the user through:

1. **Confirm the triple-axis env preconditions.** Per the
   env-precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   `pkg-config --modversion doca-pcc` returns a version the
   installed DPACC compiler is listed as compatible with under
   the DOCA Compatibility Policy; the target BlueField is
   present and exposes its DPA processor to the host (the
   `doca_dev` enumeration succeeds AND a `doca_pcc_cap_*`
   query against its `doca_devinfo` returns supported for the
   baseline custom-PCC surface); the BlueField firmware has
   the custom-PCC slot enabled; the user / process can open
   the target `doca_dev`. The firmware-side custom-PCC slot
   check is the load-bearing one the agent MUST surface
   explicitly — it is the precondition baseline agents most
   often miss, and a disabled slot will silently look like a
   permission bug at the API surface. If ANY of these fails,
   this is an env / firmware / version problem to fix via
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)
   + [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure),
   NOT a code change in the host-side PCC program.
2. **Walk the two-side-program model BEFORE writing any host
   code.** Per the two-side-program rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the user is writing TWO translation units: a host-side
   program that calls `doca_pcc_*` (load / parameterize /
   start / observe), and a DPA-side program (the
   congestion-control algorithm body) that `dpacc` compiles
   into a binary the host executable embeds as a
   `doca_pcc_app`. The agent must surface this model
   EXPLICITLY before any host-side load call is drafted. If
   the user is asking *"how do I design the algorithm
   itself"*, that is the public DOCA PCC programming guide +
   the user's domain expertise — route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the public guide; do not invent an algorithm body.
3. **Run triple-axis capability discovery against the target
   BlueField.** Per the triple-axis rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   on the DOCA side, run the matching `doca_pcc_cap_*` against
   the active `doca_devinfo` for the BlueField port the host
   is driving; on the install side, confirm `pkg-config
   --modversion doca-pcc` agrees with `doca_caps --version`
   and with the installed `dpacc` version per the DOCA
   Compatibility Policy; on the firmware side, confirm the
   custom-PCC slot is enabled. Quote ALL THREE results back
   to the user. If any says *not supported* / *not enabled*,
   that axis is the answer — do not proceed.
4. **Create the `doca_pcc` Core context against the chosen
   `doca_dev`.** Pick the `doca_dev` deliberately: it must
   map to the BlueField port whose RDMA / RoCE traffic the
   custom algorithm is meant to control. A host driving
   custom PCC on more than one port needs one `doca_pcc` per
   port, not a *"global"* one, per the per-instance rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   This is a standard DOCA Core context create — the
   universal lifecycle from
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
   applies.
5. **Load the PCC algorithm image and parameterize the
   algorithm.** Load the `doca_pcc_app` produced by `dpacc`
   from the user's DPA-side algorithm source into the
   `doca_pcc`; then set whatever host-side parameters the
   algorithm exposes. Start the `doca_pcc` via
   `doca_pcc_start()` (the `doca_pcc` has its own lifecycle
   calls — `doca_pcc_create` / `_start` / `_stop` / `_destroy` —
   and is NOT a `doca_ctx`; there is no `doca_pcc_as_ctx`) — this
   is the moment the algorithm becomes live on the port. Per the lifecycle ordering in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   record the order so teardown happens in reverse: stop the
   `doca_pcc` → release the loaded `doca_pcc_app` → destroy
   the `doca_pcc` → close the `doca_dev`.
6. **Confirm the attached port actually has RDMA / RoCE
   traffic to control.** A custom PCC algorithm on a port
   with no RDMA / RoCE flows has nothing to do — *"my
   algorithm loaded but I see no effect"* is most often
   *"there is no traffic to affect"*. If the user has not
   stood up RDMA / RoCE on this port, route to
   [`doca-rdma`](../doca-rdma/SKILL.md) BEFORE expecting
   visible PCC behavior; PCC modulates existing traffic.
7. **Sanity check before the first run.** Confirm with the
   user: which BlueField port (which `doca_dev`) the custom
   PCC algorithm is attached to; which PCC algorithm image
   (`doca_pcc_app`) is loaded and which entry points it
   exposes; which host-side parameters were set and what the
   DPA-side algorithm advertises as their valid ranges; how
   the agent will know the algorithm is having a visible
   effect on RDMA / RoCE traffic (which counter / which
   report). If any of those are unclear, stop and ask — do
   not invent.

For the canonical DOCA universal lifecycle that underlies
steps 4-5, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the PCC overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a PCC-using consumer (host-side C / C++ + at
least one DPA-side translation unit containing the custom
algorithm) against the user's installed DOCA + DPACC compiler,
with `pkg-config` + `dpacc` as the joint sources of truth for
include + link flags + DPA binary embedding.

The build pattern for any DOCA C / C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
PCC adds a DPACC layer on top — the same DPACC layer
[`doca-dpa TASKS.md ## build`](../doca-dpa/TASKS.md#build)
documents for DOCA DPA — because the custom PCC algorithm *is*
DPA-side code. The DPA-side translation unit is compiled by
`dpacc` into a binary embedded into the host executable, and
the link line on the host side pulls in the DOCA PCC library.
This skill carries only the PCC-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-pcc` | The library's `.pc` file installed by the DOCA host packages; gives the host-side include + link flags for the DOCA PCC host API |
| DPA-side toolchain | `dpacc` (DPACC compiler), installed alongside DOCA | Compiles DPA-side congestion-control algorithm translation units into the binary that the host executable embeds as the `doca_pcc_app`. The host system compiler is NOT a substitute |
| Include flags | `pkg-config --cflags doca-pcc` for the host side; the DPACC-prescribed include path for the DPA side | Resolves DOCA headers under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the include path — it can move across DOCA install profiles) for the host side; the DPA-side include path comes from the DPACC install layout via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| Link flags | `pkg-config --libs doca-pcc` on the host side, plus the DPACC-prescribed embed step that bakes the DPA-side algorithm binary into the host executable | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) on the host side. The DPACC embed step is the load-bearing build action — without it the host has no PCC algorithm image to load at runtime |
| Companion DOCA libs | `doca-argp` for arg parsing in samples | The shipped samples include both host-side and DPA-side translation units; do not invent a one-side-only manifest |
| What NOT to add to the host link line | The DPA-side libraries that the algorithm body may call from inside the DPA (named generically here so we don't invent the symbol set) — those are linked into the DPA-side translation unit by `dpacc`, NOT into the host executable | Adding DPA-side libraries to the host link line is a common first-build error; the host side links only against `doca-pcc` + `doca-common` |
| Minimum DOCA version | Query with `pkg-config --modversion doca-pcc`; never hardcode | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |
| Minimum DPACC version | Cross-check the installed `dpacc` version against the DOCA Compatibility Policy linked from [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) | Mismatched DOCA + DPACC combos fail at link time or at load time with `DOCA_ERROR_DRIVER` |

For non-C host-side consumers (Rust, Go, Python) that drive
host-side PCC setup and embed a custom PCC algorithm image
built separately by `dpacc`, the host-side link line and
version rules above still apply; the DPA-side build is a
separate compilation unit and is out of scope for this skill,
but the `dpacc` version check is the load-bearing input the
wrapper still needs.

## modify

Goal: take the closest-fitting shipped DOCA PCC sample as the
verified starting point and apply a **minimum diff** to make
it match the user's intent, without rewriting from scratch and
without inventing algorithm bodies.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the PCC-specific slot fill.

| Slot | What the agent asks the user | PCC-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_pcc/`? | Pick a sample whose **shape** matches the user's intent: same host-side parameter shape, same observability pattern, same DPA-side algorithm-entry-point set. PCC samples are two-side programs; the sample's DPA-side translation unit is the second half of the verified base, and the algorithm body in it is the starting point for the user's own algorithm |
| 2. DPA-side algorithm body | What is the algorithm actually computing? Is it a brand-new design or a tweak on the sample's algorithm? | The DPA-side translation unit is the in-place edit point for the algorithm body. The agent's anti-pattern alerts: (a) do NOT propose moving the algorithm logic to the host side — that defeats the entire reason to use PCC (the algorithm has to run per packet / per event on the DPA, not host-round-trip per packet); (b) do NOT invent a brand-new algorithm body in this skill — congestion-control algorithm design is a research / domain question, not an API question, route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| 3. Host-side parameter shape | What parameters does the algorithm take, what shapes / sizes / ranges? | Per the two-side-program rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), the host-side parameter calls and the DPA-side algorithm's parameter shape MUST agree on count, size, type, and range. Any change to one side requires updating the other; track this as a single edit, not two |
| 4. Attach-to-port choice | Which `doca_dev` will the modified algorithm attach to? Is it the same port the sample assumes? | The `doca_dev` selection in the sample's host-side configure step is what determines which BlueField port the algorithm modulates. Changing it changes which RDMA / RoCE traffic the algorithm affects; the agent must surface this even when it looks like a trivial config tweak |
| 5. Observability hook-up | How does the host observe what the running algorithm is doing? | Whatever observability the algorithm emits to the host through the loaded image's reporting surface is the user's primary signal during testing. A modify pass that changes the algorithm body without updating the observability hook-up is a common way to lose visibility silently |
| 6. Rebuild BOTH sides | After any modify, rebuild the DPA-side image via `dpacc` AND rebuild the host executable that embeds it | Per the *do not partial-rebuild one side* rule in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), rebuilding only one side is the canonical way to introduce `DOCA_ERROR_DRIVER` or silent on-wire misbehavior at load time |
| 7. Re-validate against capabilities | Re-run the `doca_pcc_cap_*` queries from [`## configure`](#configure) step 3 against the modified configuration — algorithm-feature usage growth, parameter-range growth, or attaching to a different port can flip a capability boundary | Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap query is the runtime authority |

The agent emits an *intent description + the seven filled
slots*; the *actual* unified diff against the sample source
is produced the way every other library skill in this bundle
handles modify — the agent walks the user through the diff
line-by-line against the sample source they read on disk, and
has the user paste back the result for validation. The
agent's anti-pattern alert: a *"clean rewrite"* from scratch
is almost always slower to first green than a minimum-diff
modify on a shipped PCC sample, and removes the user's
ability to bisect against a known-good baseline.

## run

Goal: actually start the loaded custom PCC algorithm on the
attached BlueField port and confirm the host observes the
algorithm reporting and the RDMA / RoCE traffic on the port
showing the expected congestion behavior.

Steps the agent should walk the user through:

1. **Confirm the loaded image exposes the algorithm entry
   points the host configure step expects.** The host's view
   of *"which algorithm is running on this port"* is exactly
   the entry-point set in the loaded `doca_pcc_app`. If the
   host-side configure step assumes entry points `dpacc` did
   not compile in, that is a build-side bug, not a runtime
   bug; route back to [`## build`](#build).
2. **Start the `doca_pcc` via `doca_pcc_start()`.** Per the
   attach-to-port section in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   `doca_pcc_start()` is the moment the
   loaded algorithm becomes live on the port; before this
   point the algorithm is loaded but inert. A successful
   start does NOT yet imply the algorithm is having a
   visible effect — that needs traffic on the port AND the
   host observing the algorithm's reports.
3. **Drive the host-side `doca_pe_progress` loop** in
   parallel with the running PCC algorithm. The host-side
   reports the algorithm emits flow through the progress
   engine; a host that starts the `doca_pcc` and then blocks
   without progressing the PE will see no reports and
   conclude *"the algorithm is broken"* incorrectly. This is
   the cross-library *"PE not progressed"* failure mode
   applied to PCC.
4. **Confirm there is RDMA / RoCE traffic on the attached
   port** to observe. If the port is idle, the algorithm
   will run but have no visible effect; route to
   [`doca-rdma`](../doca-rdma/SKILL.md) for bringing up the
   traffic the algorithm is meant to modulate.
5. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   if the host-side log is silent but the algorithm seems
   inactive, reach for the `pcc_counters` CLI named in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   and routed via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
   to inspect counters at the port — this catches the
   *"algorithm loaded fine but is not actually affecting
   traffic"* case the host-side surface alone may not show.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured custom PCC deployment actually
loads the algorithm onto the BlueField, attaches it to the
intended port, and produces visible, expected changes in the
RDMA / RoCE traffic the port carries on the user's hardware,
and that the env-precondition + capability set was sized right.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the env-precondition set, the capability set,
the two-side-program parameter shape, the algorithm-body
behavior, or the observability hook-up. The loop terminates
when either (a) the host observes the custom algorithm
affecting RDMA / RoCE traffic on the attached port the way
the user intended, or (b) the agent has narrowed the failure
cause to a layer outside PCC itself (algorithm-body design,
DPA-side hang, DPACC build, BlueField firmware
configuration, DOCA install) and escalated to the matching
skill.

Iteration shape:

1. **Single small-algorithm smoke.** Deploy ONE trivial
   custom PCC algorithm (a no-op rate-update, or a
   pass-through that emits one report) and confirm: the
   host-side load + start succeed; the cap-query baseline
   matches what was captured at configure; the host observes
   at least one report from the running algorithm. If yes,
   advance. If no — and the host-side log shows no error —
   the algorithm is loaded but not running, or the
   observability hook-up is wrong; route to
   [`## debug`](#debug) layer 5 and consult the
   `pcc_counters` tool. Per the safety-policy rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   DO NOT scale a broken smoke into a complex algorithm
   design.
2. **Traffic-affecting smoke.** Start the smoke algorithm
   with active RDMA / RoCE traffic on the attached port and
   confirm the algorithm's intended effect (even if trivial
   — e.g. a fixed rate cap) is visible in the PCC counters
   exposed by the `pcc_counters` CLI. Catches the *"the
   algorithm loaded but is not actually modulating
   traffic"* case; if it fails, the algorithm body has no
   effect path, or it is not actually attached to the
   traffic-carrying port — re-verify the `doca_dev`
   selection at configure time.
3. **Sustained-run loop.** Let the smoke algorithm run for a
   sustained period (minutes, not seconds) under RDMA /
   RoCE load and confirm the host continues to observe
   reports without loss and the counter tool shows
   continuous modulation. Catches observability-queue-sizing
   bugs and DPA-side algorithm slow leaks.
4. **Two-side-program parameter negative test.**
   Intentionally pass a host-side parameter outside the
   range the DPA-side algorithm advertises and confirm the
   host gets `DOCA_ERROR_INVALID_VALUE` cleanly — validates
   that the agent's earlier two-side-program parameter
   check is real, not just notional. Then restore the
   in-range parameter.
5. **Cap-query negative test.** Intentionally call a
   `doca_pcc_cap_*` axis the agent expects to be *not
   supported* on this BlueField + DOCA + firmware combo and
   confirm the reported `DOCA_ERROR_NOT_SUPPORTED` matches
   the triple capability discovery from [`## configure`](#configure)
   step 3 — validates the agent's capability-discovery
   itself is correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_PERMITTED` on a BlueField the agent expected to work | `doca_dev` access is fine on other libraries but PCC create / load fails | Firmware-side custom-PCC slot is the most likely culprit; re-run the firmware-side check; the BlueField may need a reset after the slot is flipped before the new state takes effect. Do NOT diagnose this as a host-OS permission problem first |
| `DOCA_ERROR_NOT_SUPPORTED` on a BlueField the agent expected to work | DOCA cap-query failed against the active `doca_devinfo` | The BlueField generation axis was missed (older BlueField may not expose the DPA at all); re-run the env-side BlueField check; if the BlueField truly does not expose the DPA, the answer is the hardware, not a code or DOCA upgrade |
| `DOCA_ERROR_DRIVER` on first algorithm load | DOCA + DPACC versions are skewed OR the algorithm image was built against a different DOCA install than the host runtime OR the firmware custom-PCC slot is enabled but in a transitional state | Re-run the version chain per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test); rebuild BOTH sides via `dpacc` + the host build against the matched versions; cross-check against the DOCA Compatibility Policy; if all that is clean, reset the BlueField to settle the firmware state |
| Smoke loaded and reports fire; counter tool shows no on-wire change | The algorithm body has no effect path on the actual rate-update events, OR there is no RDMA / RoCE traffic on the attached port to modulate | Walk the algorithm-body effect path with the user; confirm RDMA / RoCE traffic is actually flowing on the attached port (route to [`doca-rdma`](../doca-rdma/SKILL.md) if not) |
| Host observed N reports; report stream then stops mid-run | The host is not draining the report queue fast enough OR the DPA-side algorithm has hung in its main loop | Restructure the host loop to drain reports per batch; consult the DPA-side tooling named in [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) for stuck-DPA cases |
| `doca_ctx_stop` blocks on teardown | The algorithm body has an unbounded loop with no termination signal | The agent must surface that custom PCC algorithms, like generic DPA kernels, need a termination signal — see the [`doca-dpa TASKS.md ## modify`](../doca-dpa/TASKS.md#modify) DPA-kernel termination rule; the host cannot force-kill the DPA-side algorithm from the `doca-pcc` API |

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that
means the cause is below PCC (BlueField firmware slot,
algorithm design bug, DPACC bug, NIC firmware). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence including BOTH
the host-side DOCA log and the `pcc_counters` output.

## debug

Goal: when a `doca_pcc_*` call (or a custom PCC algorithm
running on the BlueField) returns a `DOCA_ERROR_*` or does
not make forward progress, narrow the cause to a specific
layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
PCC-specific fixes. This skill's overlay names the
PCC-specific manifestation at layers 5 (runtime), 6 (program),
and 7 (driver):

**Layer 5 (runtime) — PCC overlay.**

- A loaded custom PCC algorithm that produces no host-side
  report is *almost always* one of three things: the host is
  not progressing the PE; the DPA-side algorithm body has no
  emit path; or the observability queue attached to the
  `doca_pcc` is full and being silently dropped at the
  host's failure to drain. Confirm the env-side preconditions
  per [`## configure`](#configure) step 1 and the host-side
  progress per [`## run`](#run) step 3 before assuming the
  PCC algorithm itself is broken.
- A loaded custom PCC algorithm whose reports fire but whose
  RDMA / RoCE traffic shows no on-wire change is most often
  *"no traffic on the attached port"* (route to
  [`doca-rdma`](../doca-rdma/SKILL.md)) OR the algorithm
  body computes correctly but takes no rate-update action.
  Use the `pcc_counters` CLI (route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools))
  to inspect the port-level counters and confirm which case
  applies.
- A *"DPA-side PCC algorithm hung; cannot stop the program"*
  pattern is almost always a missing termination signal in
  the DPA-side algorithm body, per the
  [`doca-dpa TASKS.md ## modify`](../doca-dpa/TASKS.md#modify)
  DPA-kernel termination rule that PCC inherits.

**Layer 6 (program) — PCC overlay.**

- Lifecycle ordering: the loaded `doca_pcc_app` must be
  released BEFORE the `doca_pcc` itself is destroyed.
  Out-of-order returns `DOCA_ERROR_BAD_STATE` per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Two-side-program parameter mismatch: a host-side
  parameter call whose shape does not match the DPA-side
  algorithm's parameter shape returns
  `DOCA_ERROR_INVALID_VALUE`. The fix is to rebuild the
  DPA-side image via `dpacc` after re-aligning the
  parameter shapes, then rebuild the host executable; do
  not patch only one side. See the *do not partial-rebuild
  one side* rule in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Attached-port mismatch: if the user reports *"the
  algorithm runs but does not affect my flows"* and the
  `doca_pcc` was created against a `doca_dev` for a
  different BlueField port than the one carrying the user's
  RDMA / RoCE traffic, that is the bug. Re-verify the
  `doca_dev` selection at configure time against the
  user's traffic-carrying port.

**Layer 7 (driver) — PCC overlay.**

- `DOCA_ERROR_DRIVER` from `doca_pcc_*` is most often the
  PCC driver layer reporting failure to DOCA, and most
  often because DOCA + DPACC are skewed. Capture
  `pkg-config --modversion doca-pcc`, the installed `dpacc`
  version, and `doca_caps --version`; cross-check against
  the DOCA Compatibility Policy at
  <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>.
- `DOCA_ERROR_NOT_PERMITTED` from `doca_pcc` create / load,
  with `doca_dev` access otherwise fine, points at the
  BlueField firmware not having the custom-PCC slot
  enabled — this is a firmware-side fix, NOT a host-OS
  permission fix. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the firmware-side enable; the
  BlueField typically needs a reset after the slot is
  flipped before the new state takes effect.
- A BlueField in the wrong mode that does NOT expose the
  DPA surfaces as `DOCA_ERROR_NOT_SUPPORTED` at `doca_pcc`
  create time. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the env-side BlueField mode fix;
  this is NOT a code change in the program.

Once the layer is identified, route to the matching debug
verb on the matching skill: install / build / link / driver
to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug);
DPA-specific kernel-side hangs to
[`doca-dpa TASKS.md ## debug`](../doca-dpa/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as
follows so the agent does not invent guidance:

- **install.** Installing DOCA, installing the DPACC
  compiler, choosing matched versions, enabling the
  BlueField firmware custom-PCC slot, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA + DPACC are installed and matched
  and the firmware custom-PCC slot is enabled.
- **deploy.** Deploying custom-PCC-using applications at
  scale (multi-BlueField clusters, Kubernetes operator
  workflows for DPU workloads, multi-tenant PCC sharing) —
  out of scope for Phase 1 and reserved for a future
  platform skill. For single-host first-run testing, the
  right verb in this skill is `## run`; do not invent a
  "deploy" workflow.
- **algorithm design.** Designing the congestion-control
  algorithm body itself — out of scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA PCC* programming guide and to the
  user's own congestion-control domain expertise. This
  skill prescribes how to *load and attach* an algorithm
  from the host; it does not design the algorithm.
- **DPA-side algorithm programming and DPACC usage.**
  Writing the DPA-side algorithm body, DPA-side memory
  layout used inside the algorithm, DPACC compile flags,
  DPA-side debugging from inside the algorithm — out of
  scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA PCC*, *DOCA DPACC Compiler*, and
  *DOCA DPA* guides plus the *DPA Tools* umbrella. This
  skill prescribes how to *use* the DPACC-produced image
  from the host; it does not redefine how to produce it.
- **`pcc_counters` diagnostic CLI** — a separate
  artifact for read-only PCC counter inspection; route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
  This skill is for *custom algorithm load + control*, not
  for inspection.
- **Default factory PCC algorithm in ConnectX firmware** —
  not a `doca-pcc` topic. Configuration is firmware-side;
  route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA PCC** — it
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
| `pkg-config --modversion doca-pcc` | `## configure` step 1; `## build` minimum-version slot | What is the build-time DOCA PCC host-side library version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-pcc` | `## build` | What include + link flags does the host-side linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `which dpacc && dpacc --version` (or the install-tree path) | `## configure` step 1; `## build` minimum-DPACC slot | Is the DPACC compiler installed and at what version? | A version string the agent compares against the DOCA Compatibility Policy linked from [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility). Missing `dpacc` = the DPA-side algorithm cannot be built; route to [`doca-setup`](../../doca-setup/SKILL.md) |
| `doca_caps --list-devs` | `## configure` step 1; `## configure` step 3 | Which DOCA devices does the host see, and which expose a DPA processor (the hardware substrate the custom PCC algorithm runs on)? | One entry per `doca_dev` with the BlueField identity and the per-library capability flags including the DPA support axis. No DPA-capable entry = the BlueField is not present, not in the right mode, or not on a generation that exposes the DPA; route to [`doca-setup`](../../doca-setup/SKILL.md) |
| `ls /opt/mellanox/doca/samples/doca_pcc/` | `## modify` slot 1 | Which PCC samples ship in this install (both host-side AND DPA-side translation units), and which is the closest starting point? | A list of sample directories that each contain BOTH host-side and DPA-side source plus a `meson.build` that wires `dpacc` and `pkg-config doca-pcc` together |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last PCC call? | Empty or recent benign messages. Repeated mlx5 / PCC-driver / firmware errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug). Repeated *"custom PCC slot not enabled"* or similar → firmware-side fix |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 5 | What did the structured DOCA logger emit for the first failing host-side PCC call? | A trace-level line on every host-side lifecycle transition and every algorithm-load / parameter-set call. Silence after a `doca_pcc_start()` = either host PE not progressed OR algorithm body running but emitting nothing — reach for the counter tool next |
| (route via [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)) `pcc_counters` CLI — read-only PCC counter inspection at the port | `## test` step 2; `## debug` layer 5 | What are the actual PCC-related counters at the BlueField port doing right now, independent of the host-side `doca-pcc` program? | The public *DOCA Tools* umbrella documents the per-tool output; the agent's job is to NAME the existence of this tool and route the user there, not to redefine its surface. This is the load-bearing diagnostic for the *"the algorithm loaded but I see no on-wire change"* case |
| (route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) DPA-side developer tools — DPA debugger, DPA process-state inspector, DPA statistics tool | `## debug` layer 5; `## debug` layer 6 | What is the DPA processor itself doing right now, from the DPA side (for cases where the algorithm body is suspected to be stuck)? | The public *DPA Tools* umbrella documents the per-tool output; the agent's job is to NAME the existence of these tools and route the user there. These overlay the same surface [`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability) documents |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the PCC-specific rows on top.
