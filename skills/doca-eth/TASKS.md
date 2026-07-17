# DOCA Ethernet workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (capability
check → precondition check → single-packet smoke → completion
drain → loop back if the RX type, size, or steering plan
changed), not a one-shot pass — see the eval-loop overlay in
`## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the Ethernet capability surface,
RX-type taxonomy, TX submission shape, capability-query rules,
error taxonomy, observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
For the steering side that decides which packets land on which
RX queue, see [`doca-flow`](../doca-flow/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a `doca_eth_rxq` and / or `doca_eth_txq` on a
port, representor, or SF, with the user aware of the RX type
they need and the steering plan that has to be in place before
any packet flow is meaningful.

Steps the agent should walk the user through:

1. **Confirm the env preconditions BEFORE writing any code.**
   Per the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   verify three things: the user can open a `doca_dev` on the
   target port (sudo or `mlnx`-group membership; `id` to
   check); the port is up at the driver layer
   (`devlink dev show` reports `state: PORT_ACTIVE`;
   `ip link` shows the device `UP,LOWER_UP`); and the user has
   a steering plan for inbound traffic — either a DOCA Flow
   rule that will steer matching packets to this queue (route
   to [`doca-flow`](../doca-flow/SKILL.md)), or kernel-side
   promiscuous mode on the underlying interface (expedient
   first-run only). If any of the three is missing, fix it
   FIRST; an empty RX queue without these is silent, not a
   `DOCA_ERROR_*`.
2. **Pick the queue side(s).** RX, TX, or both. Per the RX / TX
   object table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   each is its own DOCA Core context with its own lifecycle.
   Sketch the call sequence per side before writing code.
3. **Run capability discovery against the active `doca_devinfo`.**
   For RX: `doca_eth_rxq_cap_get_max_burst_size` and, for each
   RX type the user is considering,
   `doca_eth_rxq_cap_is_type_supported(devinfo, type)`. For TX:
   `doca_eth_txq_cap_get_max_send_buf_list_len` and, if the
   user wants offload,
   `doca_eth_txq_cap_is_l3_chksum_offload_supported`. Quote the
   queried values back to the user; do not assume from prior
   installs.
4. **Pick the RX type per data shape.** Per the RX-type
   taxonomy in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   regular = per-packet user tasks (debug-friendly first-run);
   cyclic = pre-allocated buffer ring (line-rate, fixed MTU);
   managed-recv = library-owned buffers (hands-off first app).
   Do not pick *for* the user when the data-shape intent is
   ambiguous — ask.
5. **Configure the queue(s).** Mandatory before
   `doca_ctx_start()`: for RX,
   `doca_eth_rxq_set_type(<chosen>)` plus a burst size at or
   below the cap; for TX, scatter-gather length at or below the
   cap and a send-task configuration. Optional but commonly
   needed: `doca_eth_txq_set_*` for L3 checksum offload (only
   when the cap query said yes).
6. **Start the context(s)** via `doca_ctx_start()` per queue
   and progress the PE (`doca_pe_progress`). For RX, *no
   packets arriving* at this point is the steering-side
   precondition gap, not a code bug; revisit step 1. For TX,
   the queue is ready when start returns `DOCA_SUCCESS`.

For the canonical DOCA universal lifecycle that underlies steps
2-6, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the Ethernet overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a DOCA Ethernet-using consumer against the user's
installed DOCA, with `pkg-config` as the source of truth for
include + link flags.

The build pattern for any DOCA C/C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Ethernet-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-eth` | The library's `.pc` file installed by the DOCA host packages |
| Include flags | `pkg-config --cflags doca-eth` | Resolves to headers under $(pkg-config --variable=includedir doca-common) for the Ethernet subset |
| Link flags | `pkg-config --libs doca-eth` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) (plus device-side providers as the `.pc` declares them) |
| Companion libraries | `doca-flow` only if the consumer programs its own steering; `doca-argp` for argument parsing if the consumer uses the standard DOCA arg style; `doca-gpunetio` only if the consumer also uses GPU-initiated I/O on these same queues | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-eth`; never hardcode in build files | Cross-version build / runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the five slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take the closest-fitting shipped DOCA Ethernet sample and
apply a **minimum diff** to make it match the user's intent,
without rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the Ethernet-specific slot fill — the
five slots the agent must elicit from the user before
recommending any code-level edit:

| Slot | What the agent asks the user | Ethernet-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_eth/`? | Pick the closest in *queue side* (RX vs TX vs both) and *RX type* (regular vs cyclic vs managed-recv) to the user's intent. Do NOT bridge across both axes — a smaller diff is always safer than a re-architecture |
| 2. RX type change | Switching regular → cyclic or → managed-recv? | Re-run `doca_eth_rxq_cap_is_type_supported(devinfo, type)` for the new type; types are device-conditional |
| 3. Queue sizing | Changing burst size, scatter-gather depth, ring length? | Each one re-runs the matching `_cap_get_max_*` query; the device cap is the only authority. Quote the queried value, not a value the user remembered from another host |
| 4. Steering / promisc | Adding a flow rule, or moving from promiscuous mode to a real Flow rule? | This is a steering change, not an Ethernet change. Route to [`doca-flow`](../doca-flow/SKILL.md) for the rule body and back here only after the rule programs cleanly |
| 5. Offload flags | Enabling L3 checksum offload or other TX features? | Each gates on the matching `doca_eth_txq_cap_*` query; the cap is the answer if it says false |

Keep the build manifest unchanged: the sample's existing
`meson.build` already wires `pkg-config doca-eth`; do not switch
to a hand-rolled Makefile for *"simplicity"* — it removes the
version-check rail per [`## build`](#build).

The agent's anti-pattern alert: a *"clean rewrite"* from scratch
is almost always slower to first green than a minimum-diff
modify on a shipped sample, and removes the user's ability to
bisect against a known-good baseline.

## run

Goal: execute the built program and confirm one packet makes it
through end-to-end before scaling up to line rate.

Steps the agent should walk the user through:

1. **Confirm the steering plan is live.** Before starting the
   binary, confirm either the DOCA Flow rule that steers
   traffic to the RX queue is programmed (and the
   `doca_flow_pipe_*` calls returned `DOCA_SUCCESS`), or
   promiscuous mode is on at the kernel layer (`ip link show
   <dev>` reports `PROMISC`). Without one of these, the RX
   queue is silent regardless of how cleanly DOCA Ethernet
   came up.
2. **Capture the structured log.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the queue lifecycle and
   per-task transitions visible on first failure.
3. **Send a single-packet smoke.** One TX send-task with a
   minimum-size payload through `doca_eth_txq`, routed (via
   loopback, an external echo, or a paired host) back into the
   RX queue. Confirm one RX completion fires. The smoke
   isolates the queue path before traffic shape is added.
4. **Drive line-rate traffic only after the smoke passes.**
   Once the single-packet path is green, raise burst size and
   submit rate. A failure at this point is a sizing / drain
   problem, not a lifecycle problem.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove a DOCA Ethernet consumer is correct end-to-end, on
the user's installed DOCA + device + steering plan, before
claiming the *"build a first DOCA Ethernet app"* journey is
done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the capability set, the precondition set, the RX-type
choice, or the sizing. The loop terminates when either (a) the
user's intended packet pattern flows end-to-end at the expected
rate with the expected completions, or (b) the agent has
narrowed the failure cause to a layer outside DOCA Ethernet
itself (driver / firmware / network / steering) and escalated
to the matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_eth_rxq_cap_is_type_supported(devinfo, type)` for the
   chosen RX type and `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)`
   for the chosen scatter-gather depth, against the active
   `doca_devinfo`. If either is wrong, that is the answer; the
   user's device or DOCA version does not support the
   configuration. Update the user's intent or update the
   install.
2. **Precondition re-check.** Walk the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   again: device access, port up, traffic steered. The
   silent-RX-queue failure mode is *always* one of these
   three; do not look at code first.
3. **Single-packet smoke.** As in [`## run`](#run) step 3 — one
   TX → loopback / echo → one RX. If yes, advance. If no, walk
   the error: a TX-side `DOCA_ERROR_*` narrows to the TX queue
   or the send-task; no RX completion narrows to the steering
   plan or the RX type.
4. **Burst-rate smoke.** Once single-packet passes, raise rate
   to a moderate burst (e.g. 1 000 packets) and confirm every
   send-task has a matching completion and every expected RX
   slot has a recv event. Catches missing-progress bugs and
   queue-sizing mismatches.
5. **Negative test.** Once the positive path works,
   intentionally request something the device should NOT
   support (per step 1 the cap query returns false) and confirm
   the failure is the expected `DOCA_ERROR_NOT_SUPPORTED`.
   This validates the agent's capability-discovery is itself
   correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Single-packet RX never fires | Queue started cleanly, no `DOCA_ERROR_*`, no completion event | Almost always a steering / promiscuous-mode gap. Verify a DOCA Flow rule is programmed for this queue, or `PROMISC` is on the underlying interface; if both are claimed-but-true, capture `dmesg` and route to driver layer |
| TX send-task returns `DOCA_ERROR_AGAIN` after N submits | First N submissions succeed; then `AGAIN` | The queue is full; the PE is not being progressed at the rate the user is submitting. Drain completions via `doca_pe_progress()` before re-submitting, or raise queue sizing within the cap |
| RX type set returns `DOCA_ERROR_NOT_SUPPORTED` at start | A `doca_eth_rxq_cap_is_type_supported` returned `true` for the type, but `doca_ctx_start()` rejects it | The cap query was run against the wrong `doca_devinfo`, or another setter on the queue is incompatible with the chosen type. Re-narrow to the per-`devinfo` query and the full setter sequence |
| L3 checksum offload silently does not apply | Packets go out but the checksum is wrong | The user requested the offload without first calling `doca_eth_txq_cap_is_l3_chksum_offload_supported`. The fix is to cap-query first and refuse to enable the flag when false |
| Same code works on host A, fails on host B | One host advertises an RX type or offload the other does not | Re-run [`## configure`](#configure) step 3 (capability discovery) + [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on host B; the cap surface is per-device, not per-bundle |

Loop termination: stop iterating once two consecutive iterations
do not change the picture — the cause is below DOCA Ethernet
(driver / firmware / network steering / env). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured trace + capability snapshot as evidence.

## debug

Goal: when a DOCA Ethernet session fails — whether by
`DOCA_ERROR_*` return, silent RX, or sizing mismatch — isolate
the cause to a single layer before recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver. This skill provides the Ethernet
overlay at the *runtime* and *program* layers.

**Layer 5 (runtime) — Ethernet overlay.**

- Silent RX queue (no `DOCA_ERROR_*`, no completions) is
  *almost always* a steering / port-state precondition gap, not
  an Ethernet code issue. Confirm the env-side preconditions
  per [`## configure`](#configure) step 1 before any code
  change. Route the steering side to
  [`doca-flow`](../doca-flow/SKILL.md).
- `DOCA_ERROR_AGAIN` on TX submit is *always* a missing
  `doca_pe_progress()` (or insufficient drain rate vs submit
  rate). Do not recommend a retry loop; recommend a progress
  call.
- Mixed-context lifecycle bugs: RX and TX queues are
  independent contexts; calling `doca_ctx_start()` on one and
  expecting the other to be ready returns
  `DOCA_ERROR_BAD_STATE` from the un-started side. The fix is
  at configure time, not in the Ethernet call.

**Layer 6 (program) — Ethernet overlay.**

- RX-type / cap-query mismatch: requesting an RX type the
  device does not advertise returns `DOCA_ERROR_NOT_SUPPORTED`
  at the matching setter or at `doca_ctx_start()`. Walk
  the cap-query rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  before changing code.
- Lifecycle order: configure → start → use → stop → destroy,
  per side. Out-of-order returns `DOCA_ERROR_BAD_STATE`. The
  most common case is destroying the underlying `doca_dev`
  before `doca_ctx_destroy()` on the queue.
- Sizing-vs-drain mismatch: `DOCA_ERROR_NO_MEMORY` on the RX
  side, `DOCA_ERROR_AGAIN` on the TX side are not hardware
  failures; they are flow-control symptoms that the agent must
  surface as such rather than paper over with retries.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
steering to
[`doca-flow ## debug`](../doca-flow/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring, port-bring-up — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed and the port is
  up.
- **steer.** Programming the steering rules that decide which
  packets reach which RX queue (Flow pipes, RSS, hairpin,
  mirroring, conntrack) — owned by
  [`doca-flow`](../doca-flow/SKILL.md). This skill *consumes*
  Flow's steering output; it does not program it.
- **deploy.** Deploying DOCA-Ethernet-using applications at
  scale across many hosts / DPUs, Kubernetes operator workflows
  for line-rate data planes, multi-tenant queue isolation —
  out of scope for Phase 1 and reserved for a future platform
  skill.
- **firmware burn / reset / kernel driver install.** Installing
  the `mlx5_core` driver, burning new ConnectX firmware, or
  modifying `mlxconfig` parameters that need a reset — out of
  scope. Route to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver), then to the upstream MLNX OFED / firmware
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA Ethernet** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance. Run-as user is the
unprivileged user unless noted. Sudo is called out per row.

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
| `pkg-config --modversion doca-eth` | `## configure` step 1; `## build` slot 1 | What is the build-time DOCA Ethernet version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-eth` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/doca_eth/` | `## modify` slot 1 | Which DOCA Ethernet samples ship in this install, and which is the closest starting point? | A list of sample directories named after the queue + RX-type pattern they demonstrate |
| `devlink dev show` (sudo) | `## configure` step 1; `## debug` layer 7 | Is the underlying port up at the driver layer? | One row per port with `state: PORT_ACTIVE`; anything else means the port is down and the queue will be silent |
| `ip -j link show <dev>` | `## configure` step 1; `## debug` layer 7 | Does the kernel report this device as UP, and is `PROMISC` set? | `flags` contains `UP,LOWER_UP`; `promiscuity` field reflects whether promiscuous mode is on |
| `ethtool <dev>` | `## configure` step 1; `## debug` layer 7 | Does the driver report link / speed / duplex? | A non-zero speed and `Link detected: yes` |
| `doca_caps --list-devs` | `## configure` step 3 | Which devices on this host can be used as a `doca_dev` for Ethernet I/O? | One row per visible device with PCIe address and capability flags |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last Ethernet call? | Empty or recent benign messages. Repeated `mlx5` errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `mlxconfig -d <pcie> q | head -n 40` (sudo) | `## debug` layer 7 | What firmware config does the underlying NIC report? | Stable firmware config; transient values indicate a partial reset |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 2 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Ethernet-specific rows on top.
