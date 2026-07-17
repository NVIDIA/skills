# DOCA Comm Channel Admin Tool — Tasks

> **CRITICAL — body-interpretation banner.** This file repeatedly
> says *"drain"*, *"restart"*, *"per-channel inspect"*,
> *"state-changing operation"*, *"list-first → inspect-one →
> drain-or-restart"*, *"smoke-before-bulk"*, including in the
> Command appendix table rows. Those names are the **operator's
> mental model**; they are NOT subcommands or flags the
> `doca_comm_channel_admin` binary ships. The binary registers ZERO
> application-level arguments. Whenever a workflow step below says
> *"drain a channel"* or *"restart it"*, the operator does that
> through [`doca-comch`](../../libs/doca-comch/SKILL.md) (program-side
> reconnect lifecycle: `doca_ctx_stop()` → reset →
> `doca_ctx_start()`), [`doca-setup`](../../doca-setup/SKILL.md) +
> [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) (driver
> reload, BlueField-mode flip), or RShim/BFB (deepest reset) — NEVER
> by re-running the admin binary with a flag that does not exist.
> The Command appendix rows for *"Drain a stuck channel"* and
> *"Restart a stuck channel"* describe **what the operator does next
> via those other paths**, not a subcommand of this tool. The
> [`SKILL.md`](SKILL.md) top-of-file banner is the authoritative
> surface contract; this file's flow language is the operator
> overlay on top of it.

**Where to start:** The verbs that carry real workflow content are
`## run`, `## test`, and `## debug`. The other three (`configure`,
`build`, `modify`) are documented routing stubs that exist because
the bundle's verb contract is uniform. The `## test` verb is the
smoke-before-bulk loop, not a one-shot pass — see the eval-loop
overlay in `## test` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the six task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), explicitly
defers task verbs that do not belong here, and ends with the
`Command appendix` honoring the bundle's
[`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
preamble.

For the Comm Channel Admin Tool, the verbs that carry real workflow
content are `run`, `test`, and `debug`. The other three verbs
*exist as anchors* because the agent's task-verb contract is
uniform across libraries, services, and tools — and each one
carries a meaningful **routing stub** that names where the user's
question really belongs.

## configure

The Comm Channel Admin Tool takes **no configuration of its own**.
It is a flag-driven CLI that operates against the channels the
[`doca-comch`](../../libs/doca-comch/SKILL.md) library has already
created; there are no admin-tool config files, no daemons, no
environment knobs the public guide documents as required for the
tool itself.

If the user is asking *"how do I configure the Comm Channel Admin
Tool?"*, the question they almost certainly mean is one of:

- *"How do I install DOCA so the admin tool shows up?"* → route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure). The
  binary is shipped under `/opt/mellanox/doca/tools/` on installs
  that include the Comm Channel tooling subpackage; configuring is
  install.
- *"How do I configure the comch channel the admin tool will
  inspect?"* → not an admin-tool question. The channel is
  configured by the program that calls the comch library; route to
  [`doca-comch TASKS.md ## configure`](../../libs/doca-comch/TASKS.md#configure).
- *"How do I make the admin tool see a specific channel?"* → not a
  configuration question; it is a discovery question. Run the
  list operation first (see [`## run`](#run)) and let the
  channel-discovery layer in [`## debug`](#debug) guide the next
  step.

Do not invent configuration files or environment variables for
this tool. If the public guide does not document a config knob, it
does not exist.

## build

The Comm Channel Admin Tool is **shipped pre-built** as part of
every DOCA install that includes the Comm Channel tooling
subpackage. There is no source tree the external user is expected
to compile, no build flags, no `meson` or `make` workflow.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no. Route
  to [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
  The fix is to install (or re-install) DOCA with the right
  package profile, or use the public NGC DOCA container per
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install).
- *"I want to build my own admin tool that talks to comch channels
  programmatically"* → not an admin-tool question. Route to
  [`doca-comch`](../../libs/doca-comch/SKILL.md); the library's
  own capability-query family is the right programmatic surface,
  and the admin tool is a thin CLI wrapper around documented
  channel state, not a replacement for the library.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) explicitly forbids adding a build recipe
for the admin tool or shipping wrappers around it; revisit that
policy before changing this section.

## modify

**Do not modify the shipped Comm Channel Admin Tool binary.** It
is an NVIDIA-shipped CLI; there is no documented public way to
change its behavior, output format, or operation surface, and none
should be invented.

Routing for nearby "modify" questions:

- *"The output format is inconvenient — can I change it?"* → no,
  not inside this skill. The documented surface is the surface.
  If the user wants structured output, the right answer is *"check
  whether the installed version exposes one per `--help`, otherwise
  write a parser against the documented format on your installed
  version"* — and even the parser is out of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"I need different *information* than the admin tool reports"* →
  route to [`doca-comch`](../../libs/doca-comch/SKILL.md) — the
  programmatic comch capability and state surface is broader than
  what the CLI exposes for routine operation.
- *"Can I patch the tool to add a flag?"* → out of scope; this
  skill is for consumers of the shipped tool, not contributors to
  it.

## run

The Comm Channel Admin Tool exposes two functionally distinct
families of operations per
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
read-only enumeration / inspection, and state-changing
drain / restart. The flow the agent should walk the user through
when the user asks *"what's going on with this comch channel"*:

1. **Confirm the binary is present** under
   `/opt/mellanox/doca/tools/`. If absent, route to
   [`## configure`](#configure) above.
2. **List the comch channels visible from this side first.** This
   is the read-only entry point and the only safe first step. Run
   it on the side (host or BlueField Arm) where the user reports
   the problem. Capture the full output verbatim — it is the
   identifier set the `inspect` step needs.
3. **Inspect the specific channel the user is asking about.** Use
   the channel identifier from step 2; do not guess one. The
   inspection output is the per-channel state surface from
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   and is the evidence the rest of the workflow consumes.
4. **Stop here unless step 3 surfaces a stuck-state finding.** If
   the channel reads as healthy, the user's reported symptom is
   somewhere else (the program side, the driver, the network); do
   NOT escalate to drain or restart on a healthy channel. Route
   the next step to
   [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug)
   or to [`## debug`](#debug) below depending on which layer the
   evidence points at.
5. **For the exact subcommand inventory, flag names, channel
   identifier shape, and output column names** read `--help` on the
   installed version and the public DOCA Comm Channel Admin Tool
   guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
   Do **not** invent any of these from generic CLI knowledge —
   the public guide and `--help` are the joint source of truth,
   see
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

When recording the run for downstream consumers, write down: the
DOCA version (per [`doca-version`](../../doca-version/SKILL.md)),
the side the tool was run on (host vs BlueField Arm), the exact
command line used, and the full unredacted output. The downstream
`## test` and `## debug` workflows depend on those four fields.

## test

The Comm Channel Admin Tool is **the canonical smoke-before-bulk
surface** for any operation that might intervene on a comch
channel. *"Test"* in this skill means *"confirm the channel is
healthy before any state-changing action"*, not *"unit-test the
tool"*.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation — a drain, a restart, a program-side reconnect, a driver
reload, a BlueField mode change — re-opens the smoke. Treating it
as a one-shot pass is the failure mode this loop replaces.

The smoke-before-bulk shape:

1. **List one channel.** Run the read-only enumeration per
   [`## run`](#run) step 2 and pick the channel the user is about
   to touch. The list step is also the cheapest confirmation that
   the tool itself can bind to the DOCA device on this side.
2. **Inspect that channel and confirm HEALTHY.** Healthy means the
   per-channel state from
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   shows a connection state the public guide documents as steady
   for an active channel, with peer identity reported and no
   stuck-state indicators. Quote the state line; do not paraphrase.
3. **Cross-check against the program side.** The program-side
   connection callback in
   [`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability)
   must agree the channel is CONNECTED before either side is
   declared healthy. A one-sided "healthy" reading is half the
   evidence; the cross-check is what closes it.
4. **Only after steps 1-3 read clean** may the agent proceed to a
   state-changing operation (drain or restart) per
   [`## debug`](#debug) layer 4. If step 1, 2, or 3 surfaces a
   finding, the agent walks the debug ladder instead — drain or
   restart on an unhealthy channel without a root-cause is a guess.

Eval-loop overlay (rows apply to every comch deployment, not just
one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 2 → ## debug | Inspection surfaces a stuck-state finding; walk the debug ladder, then re-run step 1 | [`## debug`](#debug) |
| 1 → 2 → 3 → drain → 1 | After a drain, re-run the smoke to confirm the channel either recovered or remains stuck for a different reason | [`## debug`](#debug) layer 4 |
| 1 → 2 → 3 → restart → 1 | After a restart, re-run the smoke to confirm a fresh channel came up cleanly | [`## debug`](#debug) layer 4 |
| 1 → driver / firmware / mode change → 1 | After a driver reload or BlueField mode change, the channel view may have changed; re-run step 1 | [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| 1 (clean) → save → debug session | Once clean, the inspection is saved and consumed by the cross-cutting debug ladder | [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) |

The agent's rule: every state-changing action on the channel
re-opens the smoke. Saving a stale inspection from before a
mutation is exactly the failure mode this loop is here to prevent.

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected output is install-, version-, and
channel-state-specific; pinning one would mislead operators on a
different platform / version. See
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## debug

When the user reports a stuck channel, or when the smoke in
[`## test`](#test) does not read clean, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order. The shape of the diagnosis:

1. **Tool-not-installed.** The admin-tool binary does not exist
   under `/opt/mellanox/doca/tools/`. Confirm DOCA is installed
   (e.g. `pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`) and that the
   install profile included the Comm Channel tooling subpackage.
   Route to [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   if not.
2. **Device-binding layer.** The tool runs but cannot bind to a
   DOCA device on the side it was invoked from. Confirm the
   driver stack (`mlx5_core`, IB stack) is loaded and the
   BlueField mode is compatible with the comch transport the user
   expects. The tool's own message and `dmesg` are ground truth;
   do not guess. Route to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer
   *Driver*.
3. **Channel-discovery layer.** The tool runs, binds to a DOCA
   device, but the expected channel is absent from the listing.
   Re-run scoped to the other DOCA device on this side, confirm
   the program-side connection callback has fired CONNECTED per
   [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug),
   and confirm no previous state-changing operation already
   destroyed the channel. The right answer when the program-side
   never reached CONNECTED is to fix the program / env, not to
   keep re-running the list.
4. **Channel-state-stuck layer.** The channel is listed, the
   state reads as stuck per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 4, and the cross-check in [`## test`](#test) step 3
   agrees the program side is also wedged. This is the layer that
   *justifies* a drain or restart. Decision rule:
   - Prefer **drain** when in-flight work might still complete on
     its own and the user can tolerate the channel remaining
     up. The program side will observe quiescing rather than a
     hard reset.
   - Prefer **restart** when drain has been tried once without
     effect, when the peer process is gone, or when the user has
     explicitly accepted a `DOCA_ERROR_CONNECTION_RESET` /
     DISCONNECTED transition on the program side per
     [`doca-comch CAPABILITIES.md ## Error taxonomy`](../../libs/doca-comch/CAPABILITIES.md#error-taxonomy).
   - After **either** state-changing operation, re-run the smoke
     in [`## test`](#test) — do not chain state-changing
     operations without a fresh inspection in between.
5. **Permission layer.** Tool runs but reports it cannot read or
   change the channel because of insufficient privileges. The
   tool's own message is ground truth; re-run with the privileges
   the public guide names per
   [`doca-comch CAPABILITIES.md ## Safety policy`](../../libs/doca-comch/CAPABILITIES.md#safety-policy)
   permission matrix. Bypassing the privilege check is not on the
   table.
6. **Version layer.** Tool runs but its view of the channel
   disagrees with what the program-side `doca_comch_*` API
   reports. Walk the four-way match per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 and apply the admin-tool overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
   When the tool and the `*.so` came from different installs, the
   fix is a consistent reinstall, not a code change.
7. **Cross-cutting layer.** All layers above are clean and
   drain / restart did not resolve the symptom. The cause is
   below the comch layer — driver, firmware, BlueField mode, or
   hardware. Escalate to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured admin-tool inspection plus the program-side
   trace as evidence. Looping on drain / restart at this layer is
   the wrong move.

In every case: **quote what the tool said.** Do not paraphrase
channel-state output, do not reorder fields, do not summarize
into prose. The whole point of inspecting a channel before
touching it is to break the agent out of the
inference-from-symptom trap.

## Deferred task verbs

The four verbs below are not Comm Channel Admin Tool work and
should be routed out before the agent does any of them under this
skill's name.

- **install** ⇒ [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  (and [`## no-install`](../../doca-setup/TASKS.md#no-install) for
  the public NGC DOCA container path). The admin tool is shipped
  by the install; this skill does not own the install workflow.
- **write a Comch program** (any language) ⇒
  [`doca-comch`](../../libs/doca-comch/SKILL.md), layered on
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  The admin tool inspects channels the library created; it is not
  a template for creating them.
- **library-internal capability or state check** (e.g. per-message
  task-completion-callback semantics, producer / consumer queue
  sizing) ⇒
  [`doca-comch`](../../libs/doca-comch/SKILL.md). The admin tool
  only exposes the documented channel-state surface; deeper
  per-API state belongs to the library.
- **streaming telemetry / live metrics** ⇒ not an admin-tool
  feature. The DOCA Telemetry Service (DTS) is the documented
  telemetry surface; routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

Comm Channel Admin Tool-specific invocations the verbs above reach
for. Every row is a CLASS — the agent must not invent flags,
subcommand names, or output columns beyond `--help` on the
installed binary. The read-only-first / state-changing-after-smoke
symmetry is the load-bearing piece.

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

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover available subcommands and flags | The admin tool's own `--help` (subcommand-and-flag inventory comes from here, not from prose) | [`## run`](#run) step 5 | Prints the documented inventory; the agent uses this as the only source of truth for subcommand and flag names. |
| Confirm tool version against the comch library | The admin tool's `--version`, cross-checked with `pkg-config --modversion doca-comch` | [`## test`](#test) cross-check + [`## debug`](#debug) layer 6 | Both strings agree; disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). |
| Enumerate comch channels on this side | The admin tool's documented list-channels subcommand, run on the side the user reports the problem | [`## run`](#run) step 2 + [`## test`](#test) step 1 | Exit 0; the listing includes the channel identifier the user expected, or surfaces an empty listing the agent treats as a channel-discovery finding per [`## debug`](#debug) layer 3. |
| Inspect one channel by identifier | The admin tool's documented inspect-channel subcommand, given the identifier from the enumeration | [`## run`](#run) step 3 + [`## test`](#test) step 2 | Exit 0; the per-channel state reads as the public guide documents for a healthy active channel. |
| Save a channel-state snapshot for debug | Redirect the inspection output to a file (`> chan-state.txt`) | [`## test`](#test) "save" step + [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) | The saved file is consumed by the cross-cutting debug ladder as the channel-state half of the evidence pair. |
| Drain a stuck channel (state-changing) | **No drain subcommand exists on this binary** — this admin tool only enumerates/inspects; perform the drain through `doca-comch` itself (program-side), not via this tool | [`## debug`](#debug) layer 4 (after smoke) | The post-drain re-inspection reads as quiesced or recovered; never chained without a re-inspection per [`## test`](#test) eval loop. |
| Restart a stuck channel (state-changing) | **No restart subcommand exists on this binary** — this admin tool only enumerates/inspects; perform the restart through `doca-comch` itself (program-side), not via this tool | [`## debug`](#debug) layer 4 (after smoke; after drain did not resolve) | The post-restart re-inspection shows a fresh channel that the program side observes as a new CONNECTED transition per [`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability). |
| Re-confirm after a state-changing operation | Any of the read-only rows above, re-run after drain / restart / driver / firmware / mode change | [`## test`](#test) eval loop | The post-change output reflects the change; a stale inspection is the failure mode. |

Three cross-cutting rules for this appendix:

- **Never invent a subcommand, flag, or output column name.**
  `--help` on the installed binary plus the public DOCA Comm
  Channel Admin Tool guide via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
  are the joint contract; prose-derived names are the most common
  hallucination failure for this skill.
- **State-changing operations re-open the smoke.** Drain and
  restart are not retryable in place; after either, the agent
  re-runs the read-only smoke per [`## test`](#test).
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`pkg-config --modversion`, `dmesg`, `mlxconfig -d <bdf> q`)
  live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
  the env-side representor / PCIe enumeration lives in
  [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  this appendix names only Comm Channel Admin Tool-specific
  invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file, restated
here so they are visible at the point of action and not buried in
[`SKILL.md`](SKILL.md):

- The **public DOCA Comm Channel Admin Tool guide** plus the
  installed `--help` are the joint source of truth. When they
  disagree (e.g. a flag landed in a release this skill was not
  written against), the *installed* `--help` wins for the user's
  actual run.
- The **read-only operations are safe**; the **state-changing
  operations are not**. The agent must say which class an
  operation belongs to before recommending it, and must gate
  every state-changing operation on a clean inspection per
  [`## test`](#test).
- **Quote, do not paraphrase.** The channel-state output is the
  artifact downstream debug consumes; reformatting it loses
  fidelity that the rest of the bundle's procedures depend on.
- This skill **assumes a healthy DOCA install** (or the public
  NGC DOCA container) at both endpoints of the host ↔ DPU pair.
  If the install is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) before running
  anything else here. For the comch programming surface that
  created the channel, see
  [`doca-comch`](../../libs/doca-comch/SKILL.md).
