# DOCA Comch workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already past
a verb. The `## test` verb is an iterative loop (sample → first
modify → smoke test → narrow → loop back), not a one-shot pass —
see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the Comch capability surface, role split,
slow-path / fast-path tables, capability-query rules, error
taxonomy, observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal lifecycle,
the cross-library `DOCA_ERROR_*` taxonomy, the modify-a-shipped-
sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## configure

Goal: stand up a Comch channel between a host process and a DPU
process, with both sides aware of which path (slow / fast) they
will use, before any production message flows.

Steps the agent should walk the user through:

1. **Confirm the env preconditions.** The DPU side must see the
   host representor; the host side must see the BlueField PCIe
   address. Per the permission matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   walk `ls /sys/class/net/` on the DPU and `lspci | grep
   Mellanox` on the host BEFORE any object-creation call. If
   either is missing, route to
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)
   — this is an env problem, not a code problem.
2. **Pick the role per side.** DPU = server, host = client. Per
   the role table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   Comch is asymmetric — there is no symmetric peer-to-peer mode.
   Sketch the call sequence per side before writing code.
3. **Run capability discovery against the active `doca_devinfo`.**
   For the slow-path: `doca_comch_cap_get_max_msg_size`. For
   the fast-path: `doca_comch_consumer_cap_is_supported`,
   `doca_comch_producer_cap_is_supported`. For the server: `doca_comch_cap_get_max_clients`.
   Quote the queried values back to the user; do not assume from
   prior installs.
4. **Choose the path per side.** Slow-path only, fast-path only,
   or both. Per the slow-vs-fast table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the agent's rule: bulk data → fast-path; control plane →
   slow-path; first-app → start with slow-path, add fast-path
   only after slow-path is green end-to-end.
5. **Register the connection callbacks at the right point in the
   lifecycle.** On the server, the connection callbacks are
   registered with
   `doca_comch_server_event_connection_status_changed_register`
   (server-only — the client has no connection-event register and
   tracks state via its context state / task callbacks), plus, if
   using slow-path, the recv callback. The register requires the
   context to be idle (after `_create`, before `doca_ctx_start()`);
   callbacks registered after start are silently ignored by the
   library, and the agent must surface this.
6. **Start the context** via `doca_ctx_start()` and progress the
   PE (`doca_pe_progress`) until the connection callback reports
   CONNECTED on both sides. If the callback does not fire within
   a reasonable timeout, route to [`## debug`](#debug) ladder
   step 1 (lifecycle / connection).

For the canonical DOCA universal lifecycle that underlies steps
2-6, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the Comch overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a Comch-using consumer against the user's installed
DOCA, with `pkg-config` as the source of truth for include + link
flags.

The build pattern for any DOCA C/C++ consumer is fully documented
in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Comch-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-comch` on installs ≥ 2.5; `doca-comm-channel` on installs < 2.5 (renamed; see [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)) | Wrong module name = `pkg-config: Package 'doca-comch' was not found` on an old install, or *"the API I'm reading about isn't here"* on the legacy install. |
| Include flags | `pkg-config --cflags doca-comch` | Resolves to headers under $(pkg-config --variable=includedir doca-common) for the comch subset |
| Link flags | `pkg-config --libs doca-comch` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) |
| Companion libraries | `doca-argp` for argument parsing (if the consumer uses the standard DOCA arg style); `doca-rdma` only if the consumer also uses RDMA (Comch does not require it) | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |

For non-C consumers (Rust, Go, Python), the wrapper consumes
`libdoca_comch.so` through FFI; the build-time version
visibility goes through the language's own FFI generator (e.g.
`bindgen` against the comch headers). The role-split and
capability-discovery rules still apply — the wrapper consumes a
`*.so` that has its own runtime version per
[`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).

## modify

Goal: take the closest-fitting shipped Comch sample and apply a
**minimum diff** to make it match the user's intent, without
rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the Comch-specific slot fill.

| Slot | Value | Source |
| --- | --- | --- |
| Sample tree | `/opt/mellanox/doca/samples/doca_comch/<name>/` | Confirmed by `ls /opt/mellanox/doca/samples/doca_comch/`; one subdirectory per shipped sample |
| Pick the closest sample | Match the user's intent (server vs client; slow-path vs fast-path; one-shot vs streaming) to a sample whose code shape already matches | Per the path-selection table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| Identify the modify surface | The connection callbacks; the recv-callback body (slow-path); the producer-submit body (fast-path); the message payload struct | These are the in-place edit points; do not introduce a new translation unit unless the sample is being split for clarity |
| Re-validate against capabilities | Re-run the `doca_comch_cap_*` queries from [`## configure`](#configure) step 3 against the modified configuration — message size growth, client-count growth, transport change all flip a capability boundary | Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap query is the runtime authority |
| Keep the build manifest unchanged | The sample's existing `meson.build` already wires `pkg-config doca-comch`; do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail | Per the build slot table in [`## build`](#build) |

The agent's anti-pattern alert: a *"clean rewrite"* from scratch is
almost always slower to first green than a minimum-diff modify on a
shipped sample, and removes the user's ability to bisect against a
known-good baseline.

## run

Goal: execute the built program and confirm both sides reach
CONNECTED before any production traffic.

Steps the agent should walk the user through:

1. **Start the DPU side first** (the server). It must be listening
   on the representor before the host-side client tries to
   connect. The connection callback firing as CONNECTED on the
   server side is the *"server is ready"* signal; do not infer
   from the absence of an error log.
2. **Start the host side** (the client). The client's connection
   callback firing as CONNECTED is the *"channel is up"* signal.
3. **Send a smoke message** — one slow-path send-task with a
   small payload; verify the recv callback fires on the peer with
   the expected payload. If using fast-path, send one producer
   transfer with a small batch; verify the consumer callback
   fires.
4. **Stop in reverse order.** Client first (graceful disconnect
   surfaces in the server's connection callback as DISCONNECTED),
   then server. Tearing down the server first while the client is
   still connected produces a noisy `_CONNECTION_RESET` on the
   client side that masks unrelated bugs in subsequent runs.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove a Comch consumer is correct end-to-end, on the user's
installed DOCA + device + permissions, before claiming the
*"build a first Comch app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the path being exercised, the message size, or the
connection-state assumption. The loop terminates when the user
reports a single-message smoke + a multi-message run both work
AND the connection state machine has been observed transitioning
through CONNECTED → DISCONNECTED cleanly.

Iteration shape:

1. **Single-message slow-path smoke.** Server up, client up,
   one send-task, recv callback fires. If yes, advance. If no,
   to [`## debug`](#debug) ladder.
2. **Multi-message slow-path.** Loop 100 send-tasks with
   `doca_pe_progress()` between submits; confirm every
   completion callback fires. Catches missing-progress bugs.
3. **Disconnect / reconnect.** Stop the client; confirm the
   server's connection callback fires DISCONNECTED. Re-start the
   client; confirm CONNECTED fires again. Catches connection
   state-machine assumptions that work for the first connect but
   not subsequent ones.
4. **Fast-path smoke (if used).** Single producer transfer →
   consumer callback; then 100 transfers. The fast-path can
   pass slow-path while still failing here because the consumer
   queue sizing is independent.
5. **Cross-version run** (if the user has multiple installs):
   re-run steps 1-4 on each install; quote the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
   in the report.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Single-message smoke passed; multi-message hung | Send queue depth too small OR `doca_pe_progress()` missing | Re-narrow to the progress / queue-depth axis; widen depth via `_set_max_num_tasks` |
| Slow-path passed; fast-path consumer never fires | Consumer not attached / not progressed; fast-path producer wrote into a queue with no consumer reader | Confirm `doca_comch_consumer_create` + `_progress`; do not assume slow-path patterns transfer |
| First connect succeeded; reconnect failed with `BAD_STATE` | Connection objects not torn down before re-connect | Walk the connection-event-callback teardown; the disconnected connection needs explicit destroy before the new accept |
| Producer submit returns `AGAIN` after 100 successful submits | Consumer is not draining at the same rate | This is a flow-control problem in the app, not a Comch bug; the agent must surface the rate gap, not paper over with retries |
| Same code passes on host A, fails on host B | Different DOCA version or different representor permission | Re-run [`## configure`](#configure) step 1 (env preconditions) + [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on host B |

Loop termination: stop iterating once two consecutive iterations
do not change the picture — the cause is below Comch (driver,
hardware, env). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured trace + version state as evidence.

## debug

Goal: when a Comch session fails, isolate the cause to a single
layer before recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver. This skill provides the Comch overlay
at the *runtime* and *program* layers.

**Layer 5 (runtime) — Comch overlay.**

- The connection callback never firing is *almost always* a
  representor / PCIe visibility issue, not a Comch code issue.
  Confirm the env-side preconditions per
  [`## configure`](#configure) step 1 before any code change.
- `DOCA_ERROR_AGAIN` on submit is *always* a missing
  `doca_pe_progress()` in the user's main loop. Do not recommend
  a retry loop; recommend a progress call.
- A *"server seems up; client hangs at start"* pattern is
  most often that the server-side accept slot is occupied — the
  device's per-server cap from `doca_comch_cap_get_max_clients(devinfo)`
  is reached because a previous client did not cleanly disconnect
  (leaked connection object on the server side). The cap is a
  device capability, not a user-settable knob: agent must
  enumerate live connections and prune the leaked one, not
  invent a "raise the limit" setter.

**Layer 6 (program) — Comch overlay.**

- Slow-path vs fast-path conflation: submitting a slow-path
  send-task on the producer context returns `BAD_STATE`. Walk
  the user's path-selection (per the slow-vs-fast table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes))
  before any other diagnosis.
- Lifecycle order: configure → start → CONNECTED callback → use
  → stop → destroy. Out-of-order returns `BAD_STATE`. The most
  common case is submitting before the connection callback
  reports CONNECTED.
- Message-size overflow: a slow-path send-task with a payload
  bigger than `doca_comch_cap_get_max_msg_size(devinfo)` returns
  `INVALID_VALUE`. The fix is to fragment at the app layer or
  switch to the fast-path producer; not to raise the cap (it is
  device-bound).

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying Comch-using applications at scale (many
  host processes per BlueField, Kubernetes operator workflows
  with sidecar BlueField agents) — out of scope for Phase 1 and
  reserved for a future platform skill.
- **manage / monitor.** Multi-tenant Comch fan-out, telemetry
  shipping over Comch — covered partially by the operations
  guidance in
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  no operator skill yet ships in this bundle.
- **firmware burn / reset.** Comch depends on the underlying
  ConnectX firmware and BlueField BFB; if the user's debug
  ladder lands on a driver-layer issue, the fix is via
  `mlxconfig` / `mlxfwreset` / re-imaging the BFB, all of which
  belong to the env-side skill rather than this one.

## Command appendix

Every command below is **cross-cutting on DOCA Comch** — it
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
| `pkg-config --modversion doca-comch` | `## configure` step 1; `## build` slot 1 | What is the build-time DOCA Comch version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-comch` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/doca_comch/` | `## modify` | Which Comch samples ship in this install, and which is the closest starting point? | A list of sample directories named after the role + path pattern they demonstrate |
| `ls /sys/class/net/` (DPU side) | `## configure` step 1 | Which host representors are visible to the DPU? | One entry per representor (PF / VF / SF) the host has surfaced |
| `lspci | grep Mellanox` (host side) | `## configure` step 1 | Which BlueField PCIe addresses are reachable from the host? | One row per BlueField PF, plus VFs and SFs depending on config |
| `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` (sudo) | `## configure` step 1 | What mode is this BlueField in (SmartNIC / DPU / switch)? | A single matching mode line; Comch requires the host ↔ DPU pair to agree |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last Comch call? | Empty or recent benign messages. Repeated mlx5 / IB errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Comch-specific rows on top.
