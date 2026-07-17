# DOCA Rivermax workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop
(Rivermax-precondition check → capability check → underlying
queue + steering check → single-frame smoke → completion drain
→ loop back if the stream type, frame size, or scheduling plan
changed), not a one-shot pass — see the eval-loop overlay in
`## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the Rivermax capability surface,
the receive-only input stream model, capability-query rules, error
taxonomy, observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
For the queue surface that carries the packets a Rivermax
stream produces or consumes, see
[`doca-eth`](../doca-eth/SKILL.md). For the steering side
that decides which packets land on which queue, see
[`doca-flow`](../doca-flow/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: initialize the process-global DOCA Rivermax engine
(`doca_rmax_init()`) and stand up a receive-only
`doca_rmax_in_stream` on a port, representor, or SF, with the
user aware that the Rivermax SDK +
license precondition is a gate (not an error path) and with the
underlying queue + steering plan in place before any frame flow
is meaningful.

Steps the agent should walk the user through:

1. **Confirm the Rivermax precondition AND env preconditions
   BEFORE writing any code.** Per the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   verify (in this order, because later rows are wasted effort
   if the first two fail):
   (a) the NVIDIA Rivermax SDK is installed at its expected
   location on the host;
   (b) a valid Rivermax license is readable by the user the
   streaming process will run as;
   (c) the user can open a `doca_dev` on the target port (sudo
   or `mlnx`-group membership; `id` to check);
   (d) the port is up at the driver layer (`devlink dev show`
   reports `state: PORT_ACTIVE`; `ip link` shows the device
   `UP,LOWER_UP`);
   (e) the user has a steering plan for inbound traffic — a
   DOCA Flow rule that will steer matching packets to the
   `doca_eth_rxq` the input stream will be attached to (route
   to [`doca-flow`](../doca-flow/SKILL.md)).
   If (a) or (b) is missing, **stop here** and tell the user
   honestly that `doca-rmax` cannot be used without those
   two — there is no DOCA-side fallback that preserves timing
   precision. Route the install / license question to the
   public Rivermax SDK guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
2. **Stand up the underlying packet queue first.** Per the
   queue-pair workflow in
   [`doca-eth TASKS.md ## configure`](../doca-eth/TASKS.md#configure),
   bring up a `doca_eth_rxq` on the target port (the Rivermax
   public API is receive-only). `doca-rmax` integrates *with*
   this queue; it does not replace it.
3. **Initialize the global engine and create the input stream.**
   Call `doca_rmax_init()` once for the process (after any
   `doca_rmax_set_cpu_affinity_mask()`), then create the
   receive-only `doca_rmax_in_stream` via
   `doca_rmax_in_stream_create()` on a `doca_dev` (which must
   have a valid IPv4 address) and convert it to a DOCA Core
   context with `doca_rmax_in_stream_as_ctx()`. Per the
   object-model table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   there is no per-integration context object and no
   transmit/output stream. Sketch the call sequence before
   writing code.
4. **Run capability discovery against the active `doca_devinfo`.**
   Walk the `doca_rmax_get_*_supported` family per the
   capability-query rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   for PTP-clock support and hardware-accelerated
   packet-placement order (RTP-seqn, ST 2110-20 seqn) the user
   is considering. Quote the
   queried values back to the user; do not assume from prior
   installs or from the public docs. The cap query is the
   runtime authority — when it says not-supported, that is the
   answer.
5. **Configure the input stream.** Set the scatter type,
   timestamp format, memory-block / element-count layout, and
   (only after step 4 confirmed support) the packet-placement
   order and PTP-synced timestamp via the
   `doca_rmax_in_stream_set_*` setters. Wire the underlying
   `doca_eth_rxq` in. Register the Rx-data event via
   `doca_rmax_in_stream_event_rx_data_register()`. Plan the
   real-time
   scheduling discipline for the streaming threads now,
   before start — the canonical scheduling guidance lives in
   the Rivermax SDK guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
6. **Start the input stream** via `doca_ctx_start()`, attach a
   `doca_rmax_flow` filter with `doca_rmax_flow_attach()`, and
   progress the PE (`doca_pe_progress`). *No frames
   arriving* at this point is a Rivermax-license / steering /
   underlying-queue precondition gap, not a code bug; revisit
   step 1.

For the canonical DOCA universal lifecycle that underlies
steps 2-6, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the Rivermax overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a DOCA Rivermax-using consumer against the
user's installed DOCA *and* installed Rivermax SDK, with
`pkg-config` as the source of truth for the DOCA-side include
+ link flags and the Rivermax SDK's own integration mechanism
for the Rivermax-side ones.

The build pattern for any DOCA C/C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Rivermax-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name (DOCA side) | `doca-rmax` | The library's `.pc` file installed by the DOCA host packages. A missing `.pc` does NOT imply the Rivermax SDK is missing — they are two independent install layers |
| Include flags (DOCA side) | `pkg-config --cflags doca-rmax` | Resolves to headers under $(pkg-config --variable=includedir doca-common) for the Rivermax-integration subset |
| Link flags (DOCA side) | `pkg-config --libs doca-rmax` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) (plus device-side providers as the `.pc` declares them). The DOCA-side link does NOT include the Rivermax SDK itself — that comes from the SDK's own integration mechanism |
| Rivermax SDK integration | Per the public Rivermax SDK guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | The Rivermax SDK ships its own include / link integration mechanism (header location, library path, license-file path). The agent must consult the Rivermax SDK guide rather than invent flags |
| Companion DOCA libraries | `doca-eth` (always — the queue surface is required); `doca-flow` only if the consumer programs its own steering; `doca-argp` for argument parsing if the consumer uses the standard DOCA arg style | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-rmax`; never hardcode in build files | Cross-version build / runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility); the Rivermax SDK version is a second axis the build must also honor |

For non-C consumers (Rust, Go, Python), the link surface is
the same `*.so` files; the FFI wrapper layer is the
language-specific binding and is out of scope for this skill —
but the six slots above are still the load-bearing inputs the
wrapper needs, with the Rivermax SDK integration row being the
most likely to surprise a wrapper author who assumes DOCA
bundles the Rivermax engine.

## modify

Goal: take the closest-fitting shipped DOCA Rivermax sample
and apply a **minimum diff** to make it match the user's
intent, without rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the Rivermax-specific slot fill — the six
slots the agent must elicit from the user before recommending
any code-level edit:

| Slot | What the agent asks the user | Rivermax-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_rmax/`? | Pick the closest in *stream type* (e.g. SMPTE ST 2110 video vs audio vs market data) to the user's intent. All public samples are receive-side; a smaller diff is always safer than a re-architecture |
| 2. Rivermax precondition still satisfied? | Has anything changed about the Rivermax SDK install or license since the last working build? | A working sample that suddenly stops working after a modify is sometimes really a license that expired between runs; do not assume the diff is the cause without re-checking the precondition matrix in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) |
| 3. Stream-type change | Switching the stream type (e.g. ST 2110 video → audio, or video frame rate / size change)? | Re-run the matching `doca_rmax_get_*_supported` query (PTP clock, packet-placement order) for the new configuration against the active `doca_devinfo`; capability is the joint property of device + DOCA version + Rivermax SDK version |
| 4. Underlying queue change | Changing the underlying `doca_eth_rxq` sizing, burst size, or RX type? | This is an `doca-eth` change, not a Rivermax change. Route to [`doca-eth`](../doca-eth/SKILL.md) for the queue body and back here only after the queue change re-validates against its own cap queries |
| 5. Steering / Flow rule change | Adding or modifying the Flow rule that steers traffic to the queue? | This is a steering change, not a Rivermax change. Route to [`doca-flow`](../doca-flow/SKILL.md) for the rule body and back here only after the rule programs cleanly |
| 6. Scheduling discipline | Changing the real-time priority of the streaming thread, the CPU pinning, or the isolation? | This is a Rivermax-side concern; route the canonical scheduling guidance via the public Rivermax SDK guide reachable through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). The agent must not invent scheduler flags |

Keep the build manifest unchanged: the sample's existing
`meson.build` already wires `pkg-config doca-rmax`
alongside the Rivermax SDK integration; do not switch to a
hand-rolled Makefile for *"simplicity"* — it removes both the
version-check rail per [`## build`](#build) and the Rivermax
SDK integration the sample author already got right.

The agent's anti-pattern alert: a *"clean rewrite"* from
scratch is almost always slower to first green than a
minimum-diff modify on a shipped sample, and removes the
user's ability to bisect against a known-good baseline — and
in the Rivermax case it also removes the implicit confirmation
that the Rivermax SDK integration was correct in the working
sample.

## run

Goal: execute the built program and confirm one frame makes it
through end-to-end before scaling up to the full stream rate.

Steps the agent should walk the user through:

1. **Confirm the Rivermax license is still readable.** Before
   starting the binary, confirm the Rivermax license file is
   present and readable by the user the process will run as.
   A license that expired or moved between configure and run
   is the second-most-common Rivermax first-app symptom (after
   *"I didn't install Rivermax at all"*) and presents as a
   `DOCA_ERROR_NOT_PERMITTED` that an agent without this rule
   will misread as a DOCA-side device-access shortfall.
2. **Confirm the steering plan is live.** Before starting the
   binary (input direction), confirm the DOCA Flow rule that
   steers traffic to the underlying `doca_eth_rxq` is
   programmed (and the `doca_flow_pipe_*` calls returned
   `DOCA_SUCCESS`). Without it, the Rivermax input stream
   will be silent regardless of how cleanly Rivermax came up.
3. **Set the real-time scheduling on the streaming thread.**
   Per the canonical Rivermax SDK guidance (route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)),
   the streaming thread should run at real-time priority for
   sub-microsecond jitter to hold. Confirm the program is
   actually setting this — defaults are not enough.
4. **Capture the structured log.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the per-stream lifecycle
   and per-event transitions visible on first failure.
5. **Send a single-frame smoke.** One frame / chunk of the
   configured stream type: a single matching frame routed in
   via an external sender or loopback. Confirm one Rx-data
   event fires. The smoke isolates the Rivermax receive path
   before timing-precise rate is added.
6. **Drive full stream rate only after the smoke passes.**
   Once the single-frame path is green, raise to the
   configured stream rate. A failure at this point is a
   jitter / scheduling-discipline / sizing problem, not a
   lifecycle problem.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove a DOCA Rivermax consumer is correct end-to-end,
on the user's installed DOCA + Rivermax SDK + device + license
+ steering plan, before claiming the *"build a first DOCA
Rivermax app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the precondition set, the capability set, the
stream-type choice, the underlying queue / steering plan, or
the scheduling discipline. The loop terminates when either
(a) the user's intended stream flows end-to-end at the
expected rate with the expected jitter and the expected
completions, or (b) the agent has narrowed the failure cause
to a layer outside DOCA Rivermax itself (Rivermax SDK /
license / driver / firmware / steering / scheduling) and
escalated to the matching skill or to the public Rivermax SDK
guide.

Iteration shape:

1. **Rivermax precondition re-check.** Re-walk the first two
   rows of the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   Rivermax SDK installed; license readable. If either is
   wrong, the answer is the install / license fix, not a code
   change.
2. **Capability re-check.** Re-run the matching
   `doca_rmax_get_*_supported` query for the chosen timestamp
   format / packet-placement order against the active
   `doca_devinfo`. If wrong, that is the answer; the device,
   DOCA version, or Rivermax SDK version does not support the
   configuration. Update the user's intent or update the
   install.
3. **Underlying queue + steering re-check.** Walk
   [`doca-eth TASKS.md ## test`](../doca-eth/TASKS.md#test)
   for the queue side and
   [`doca-flow TASKS.md ## test`](../doca-flow/TASKS.md#test)
   for the steering side. A silent Rivermax input stream is
   almost always one of these two.
4. **Single-frame smoke.** As in [`## run`](#run) step 5 —
   one frame in or out, one completion. If yes, advance. If
   no, walk the error: a `DOCA_ERROR_*` narrows to the
   `doca_rmax_in_stream` object or to `doca_rmax_init()`; no
   completion narrows to license, steering, queue, or
   missing-progress.
5. **Stream-rate smoke.** Once single-frame passes, raise to
   the configured full stream rate and confirm every expected
   completion event fires on time. Catches scheduling-
   discipline gaps (jitter rises) and queue-sizing
   mismatches.
6. **Negative test.** Once the positive path works,
   intentionally request something the device + Rivermax
   should NOT support (per step 2 the cap query returns
   false) and confirm the failure is the expected
   `DOCA_ERROR_NOT_SUPPORTED`. This validates the agent's
   capability-discovery is itself correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| First Rivermax create returns `DOCA_ERROR_NOT_PERMITTED` | The DOCA-side device access has been validated independently; the `_NOT_PERMITTED` is from a Rivermax call, not a plain `doca_dev_open` | First hypothesis is Rivermax license missing / expired / not readable. Re-walk the first two precondition rows; ONLY then consider DOCA-side device access. |
| Input stream started cleanly, no completions | Stream + underlying queue both STARTED; no `DOCA_ERROR_*`; no Rx-data event | Almost always (a) Flow rule missing on the steering side, or (b) Rivermax license problem the agent missed at configure, or (c) PE not progressed in the user's main loop. Walk those three in order. |
| Stream-rate smoke shows jitter / dropped frames | Single-frame smoke passed cleanly; full-rate run shows jitter past the Rivermax spec | The streaming thread is not at real-time priority, or the CPU is not isolated, or another high-priority thread is preempting it. Route the scheduling discipline to the public Rivermax SDK guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). |
| Placement-order set returns `DOCA_ERROR_NOT_SUPPORTED` at start | A `doca_rmax_get_*_supported` query returned `DOCA_SUCCESS`, but `doca_ctx_start()` rejects the configuration | The cap query was run against the wrong `doca_devinfo`, or a `doca_rmax_in_stream_set_*` setter is incompatible with the chosen configuration. Re-narrow to the per-`devinfo` query and the full setter sequence. Confirm the Rivermax SDK version on the host matches the one the cap-query result implies. |
| Same code works on host A, fails on host B | One host has the Rivermax SDK + license; the other does not, or has a different Rivermax version | Re-walk the precondition matrix on host B; then re-run [`## configure`](#configure) step 4 (capability discovery) + [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on host B. The cap surface is per-device AND per-Rivermax-version. |

Loop termination: stop iterating once two consecutive
iterations do not change the picture — the cause is below
DOCA Rivermax (Rivermax SDK internals / license server /
driver / firmware / network steering / scheduling-discipline /
env). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured trace + capability snapshot + Rivermax-side
state as evidence.

## debug

Goal: when a DOCA Rivermax session fails — whether by
`DOCA_ERROR_*` return, silent stream, or jitter-budget
violation — isolate the cause to a single layer before
recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver. This skill provides the Rivermax
overlay at the *runtime* and *program* layers, with one
additional rule that overlays at *install*: Rivermax SDK +
license is a second install axis that the standard ladder does
not check.

**Layer 1 (install) — Rivermax overlay.**

- Confirm that the Rivermax SDK is installed AND a valid
  license is readable BEFORE diagnosing anything else. This
  is the gate. A failure here cannot be papered over by any
  later layer; route to the public Rivermax SDK guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

**Layer 5 (runtime) — Rivermax overlay.**

- Silent Rivermax input stream (no `DOCA_ERROR_*`, no
  completions) is *almost always* a missing Rivermax license,
  a missing-or-wrong Flow rule, or an unstarted underlying
  `doca_eth_rxq`. Confirm those env-side preconditions per
  [`## configure`](#configure) step 1 before any code change.
- `DOCA_ERROR_NOT_PERMITTED` on a Rivermax call after the
  DOCA-side device access has been independently validated is
  the canonical *"Rivermax license missing / expired / not
  readable"* signal — not the DOCA-side `mlnx` group
  shortfall.
- Jitter past spec at full stream rate is *almost always* a
  scheduling-discipline gap — the streaming thread is not at
  real-time priority, or the CPU is not isolated, or another
  high-priority thread is preempting it. Route to the public
  Rivermax SDK guide for the canonical scheduling discipline
  via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

**Layer 6 (program) — Rivermax overlay.**

- Stream-type / cap-query mismatch: requesting a stream type
  / frame size / packet rate the device + Rivermax SDK does
  not advertise returns `DOCA_ERROR_NOT_SUPPORTED` at the
  matching setter or at `doca_ctx_start()`. Walk the
  cap-query rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  before changing code.
- Lifecycle order: configure → start → use → stop → destroy
  on the `doca_rmax_in_stream` context. Out-of-order returns
  `DOCA_ERROR_BAD_STATE`. The most
  common case is treating the process-global `doca_rmax_init()`
  call as if it started the input stream context.
- Wrong-layer responsibility: programming the queue from
  within Rivermax code, or programming steering from within
  Rivermax code, is a category error — Rivermax is the
  Rivermax integration surface, not the queue surface (use
  [`doca-eth`](../doca-eth/SKILL.md)) and not the steering
  surface (use [`doca-flow`](../doca-flow/SKILL.md)).

Once the layer is identified, route to the matching debug
verb on the matching skill: install / build / link / driver
to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
Rivermax SDK / license install to the public Rivermax SDK
guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
version to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
steering to
[`doca-flow ## debug`](../doca-flow/TASKS.md#debug);
queue to
[`doca-eth ## debug`](../doca-eth/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install (Rivermax SDK + license).** Installing the NVIDIA
  Rivermax SDK, configuring the license file, renewing /
  rotating the license — out of scope for this skill (and
  out of scope for the whole DOCA bundle). Route via the
  public Rivermax SDK guide reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill assumes the SDK + license are already present.
- **install (DOCA).** Installing DOCA, choosing packages,
  post-install verification, `pkg-config` wiring,
  port-bring-up — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
- **steer.** Programming the steering rules that decide which
  packets reach which queue (Flow pipes, RSS, hairpin,
  mirroring) — owned by
  [`doca-flow`](../doca-flow/SKILL.md). This skill
  *consumes* Flow's steering output via the underlying queue;
  it does not program it.
- **queue.** Programming the underlying packet queue
  (`doca_eth_rxq` / `doca_eth_txq`) — owned by
  [`doca-eth`](../doca-eth/SKILL.md). DOCA Rivermax
  integrates with the queue; it does not replace it.
- **deploy.** Deploying DOCA-Rivermax-using applications at
  scale across many hosts / DPUs, Kubernetes operator
  workflows for timing-precise media data planes, multi-tenant
  stream isolation — out of scope for Phase 1 and reserved for
  a future platform skill.
- **firmware burn / reset / kernel driver install.** Installing
  the `mlx5_core` driver, burning new ConnectX firmware, or
  modifying `mlxconfig` parameters that need a reset — out of
  scope. Route to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver), then to the upstream MLNX OFED / firmware
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA Rivermax** — it
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
| `pkg-config --modversion doca-rmax` | `## configure` step 1; `## build` slot 1 | What is the build-time DOCA Rivermax (wrapper) version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version ## debug`](../../doca-version/TASKS.md#debug) layer 2). Success here does NOT imply the Rivermax SDK is installed |
| `pkg-config --cflags --libs doca-rmax` | `## build` | What include + link flags does the linker need on the DOCA side? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/doca_rmax/` | `## modify` slot 1 | Which DOCA Rivermax samples ship in this install, and which is the closest starting point? | A list of receive-side sample directories named after the stream-type pattern they demonstrate. An empty result means no samples shipped on this install — route to [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the public DOCA Rivermax guide |
| `devlink dev show` (sudo) | `## configure` step 1; `## debug` layer 7 | Is the underlying port up at the driver layer? | One row per port with `state: PORT_ACTIVE`; anything else means the port is down and the Rivermax stream will be silent |
| `ip -j link show <dev>` | `## configure` step 1; `## debug` layer 7 | Does the kernel report this device as UP? | `flags` contains `UP,LOWER_UP`. Promiscuous mode is generally NOT how Rivermax input streams get their packets — they get them via a Flow rule on the underlying queue |
| `ethtool <dev>` | `## configure` step 1; `## debug` layer 7 | Does the driver report link / speed / duplex? | A non-zero speed and `Link detected: yes` |
| `doca_caps --list-devs` | `## configure` step 4 | Which devices on this host can be used as a `doca_dev` for Rivermax integration? | One row per visible device with PCIe address and capability flags |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last Rivermax call? | Empty or recent benign messages. Repeated `mlx5` errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug). Repeated Rivermax-side errors → route to the Rivermax SDK guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| `mlxconfig -d <pcie> q | head -n 40` (sudo) | `## debug` layer 7 | What firmware config does the underlying NIC report? | Stable firmware config; transient values indicate a partial reset |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 4 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every event / task. Silence after a Rivermax create = license / SDK / PE-progress problem, not a code bug |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Rivermax-specific rows on top. For the
Rivermax SDK's own diagnostic commands (license check,
Rivermax-side log capture, Rivermax-side version query) the
canonical reference is the public Rivermax SDK guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
the agent must not invent Rivermax-side commands from memory.
