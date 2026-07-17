# DOCA Comm Channel Admin Tool — Capabilities

> **CRITICAL — body-interpretation banner (read before everything
> below).** This file repeatedly says *"drain a channel"*,
> *"restart it"*, *"per-channel inspect step"*, *"state-changing
> operation gate"*, *"list-first invocation"*, and *"smoke-before-bulk"*.
> Those names describe the **operator's mental model** for what they
> are about to do AFTER reading the admin tool's two ASCII tables —
> they are NOT subcommands or flags the `doca_comm_channel_admin`
> binary ships. The binary registers ZERO application-level
> arguments (`register_comm_channel_admin_params()` returns
> `DOCA_SUCCESS` with no `doca_argp_param_create()` calls), shells
> out to `resourcedump` (MFT), and prints two read-only tables.
> Mapping from the legacy names below onto the real surface:
> *"enumerate channels"* / *"inspect channel state"* / *"list-first"*
> = the single read-only invocation; *"drain"* / *"restart"* /
> *"state-changing operation"* = operator goes to
> [`doca-comch`](../../libs/doca-comch/SKILL.md) program side (reset +
> `doca_ctx_stop()` → `doca_ctx_start()` reconnect), or
> [`doca-setup`](../../doca-setup/SKILL.md) + [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
> for driver reload, or RShim/BFB for the deepest reset; *"smoke-before-bulk"*
> = run the admin tool once to confirm the tables look healthy
> BEFORE the operator takes one of those other-path state-changing
> actions. Do not invent a `--drain` / `--restart` / `--inspect`
> flag for the admin binary based on the language below — the [`SKILL.md`](SKILL.md)
> top-of-file banner is the authoritative surface contract.

**Where to start:** The tool is a single admin CLI; the pattern
overview below names the recurring admin-tool questions. Pick the
pattern first, then drill into the H2 that owns the substance. For
the *how* of executing each pattern, jump to [TASKS.md](TASKS.md).
For the comch programming surface the channels were created
against, see
[`doca-comch CAPABILITIES.md`](../../libs/doca-comch/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
the tool reports and changes*, *what versions it ships in*, *the
layered error and observability surfaces*, and *the safety policy
that gates state-changing operations* (drain, restart) behind a
clean inspection. For step-by-step invocations and the
smoke-before-bulk workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every Comm Channel Admin Tool question this skill teaches resolves
into one of FIVE patterns. The patterns are CLASSES — they apply
across every DOCA install that ships the tool, not just one
platform.

| Admin-tool pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Enumerate channels | List every comch channel the tool can see so the agent can refer to real channel identifiers, not invented ones | [`## Capabilities and modes`](#capabilities-and-modes) channel-enumeration family + [TASKS.md ## run](TASKS.md#run) |
| 2. Inspect channel state | Read the per-channel state the tool reports (connection state, peer identity, queue health) before any state-changing action | [`## Capabilities and modes`](#capabilities-and-modes) channel-state family + [TASKS.md ## run](TASKS.md#run) inspect step |
| 3. Cross-check against the program | Confirm the admin tool's view of a channel agrees with the program-side `doca_comch_*` connection callback before declaring either side authoritative | [`## Observability`](#observability) cross-cutting pairing + [TASKS.md ## test](TASKS.md#test) |
| 4. Smoke-before-bulk | List → inspect one channel → confirm HEALTHY before any drain / restart on the same or another channel | [TASKS.md ## test](TASKS.md#test) + [`## Safety policy`](#safety-policy) state-changing-operation gate |
| 5. Diagnose stuck / empty / mismatched output | Map the symptom (empty list, stuck state, permission denied, version mismatch) to the right layer before any code change | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Read-only first; state-changing second.** Enumeration and
  inspection are side-effect-free; drain and restart change channel
  state and can interrupt traffic. The agent must run a
  read-only inspection before recommending any state-changing
  operation, and never re-issue a state-changing operation as a
  retry without a fresh inspection in between.
- **The admin tool is one half of the picture.** The other half is
  the program-side `doca_comch_*` connection callback per
  [`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability).
  An agent that quotes the admin tool's state without the program
  side (or vice versa) is missing half the evidence.

## Capabilities and modes

The DOCA Comm Channel Admin Tool ships as a CLI binary under
`/opt/mellanox/doca/tools/` on every DOCA install that includes the
tool. There is no daemon, no library to link against, and no
programmatic API; the user's entire interaction model is *invoke
the binary and read the two printed read-only tables*. The binary
registers ZERO application-level arguments and takes no
state-changing subcommand; drain / restart live on other paths per
the body-interpretation banner at the top of this file.

The tool exposes a single, read-only family of operations,
documented in the public DOCA Comm Channel Admin Tool guide
(reached via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)):

- **Read-only operations (the entire tool surface).** The binary
  scans every DOCA device on the side it runs on (host or
  BlueField Arm), shells out to `resourcedump` for the devices
  that support a comch client / server, and prints two tables — a
  CONNECTIONS table and a SERVERS table — naming each channel's
  server name, owning PID, PCIe address, and interface (plus the
  connected / max-connection counts for servers). These cost
  nothing to run and are the agent's default reach when the user
  reports a Comch-side issue.

The operator's *state-changing* actions — drain a channel (let
outstanding transfers complete and then quiesce) or restart it
(tear down and re-establish) — are NOT operations this binary
performs. As the body-interpretation banner at the top of this
file spells out, they live on other paths: the program-side
`doca_comch_*` reset (`doca_ctx_stop()` → `doca_ctx_start()`) per
[`doca-comch`](../../libs/doca-comch/SKILL.md), or driver /
BlueField reload per [`doca-setup`](../../doca-setup/SKILL.md) and
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md). Such
a state change can surface to the program side as
`DOCA_ERROR_CONNECTION_RESET` or a DISCONNECTED transition on the
connection callback per
[`doca-comch CAPABILITIES.md ## Error taxonomy`](../../libs/doca-comch/CAPABILITIES.md#error-taxonomy);
the admin tool's read-only scan is what the operator runs BEFORE
taking one of those other-path actions.

The tool runs on **both sides of a host ↔ DPU pair** — the host
sees host-side clients (each bound to a BlueField PCIe address),
and the BlueField Arm sees the DPU-side server (bound to a host
representor). The set of channels visible from one side is not
necessarily symmetric with the other side; the cross-checking
pattern in [`TASKS.md ## test`](TASKS.md#test) is how the agent
reconciles the two views.

The exact, current subcommand inventory, flag names, channel
identifier shape, and output column names live in the public guide
and in the tool's own `--help` on the installed version. The skill
deliberately does not pin them — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The Comm Channel Admin Tool-specific overlay** is:

- **The tool ships under the `Comm Channel` name even on DOCA installs ≥ 2.5.** The companion library was renamed to `doca-comch` in DOCA 2.5 (see [`doca-comch CAPABILITIES.md ## Version compatibility`](../../libs/doca-comch/CAPABILITIES.md#version-compatibility) for the rename); the admin tool's public guide URL slug remains `DOCA-Comm-Channel-Admin-Tool`. An agent searching for *"DOCA Comch Admin"* on `docs.nvidia.com` should fall back to the Comm Channel Admin Tool guide and surface the naming asymmetry to the user, not assume the tool is missing.
- **Confirm the tool is present before assuming availability.** If the user reports the binary is absent under `/opt/mellanox/doca/tools/`, the right answer is to confirm the installed DOCA version per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure) and route to [`doca-setup`](../../doca-setup/SKILL.md) for an upgrade or reinstall, not to recommend a wrapper script that simulates the tool from outside.
- **Where it runs:** on the x86 / Arm host that has DOCA installed, *or* on the BlueField Arm side. Same binary, same flags; the set of channels it sees differs by side per [`## Capabilities and modes`](#capabilities-and-modes).
- **Tool version must match the comch `*.so` it inspects.** A program built against one DOCA train inspected by an admin tool from a different train is the same partial-install hazard as case (a) ≠ (c) in the four-way-match rule. When in doubt, run `pkg-config --modversion doca-comch` and the tool's own `--version` per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) and quote both.

## Error taxonomy

The admin tool's error surface is broader than `doca_caps` because
it both reads channel state and (in the state-changing operations)
can mutate it. The error layers the agent should distinguish, in
escalating order:

1. **Tool-not-installed.** The admin-tool binary does not exist
   under `/opt/mellanox/doca/tools/`. Cause: DOCA is not installed
   on this host, the install does not include the Comm Channel
   tooling subpackage, or the install version pre-dates the
   tool's availability. Routing:
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and the version-compatibility overlay above.
2. **Device-binding layer.** The tool runs but cannot bind to a
   DOCA device on the side it was invoked from. Cause: the
   underlying driver stack (`mlx5_core`, IB stack) is not loaded,
   the BlueField mode is incompatible with the Comch transport
   the user expects, or — on the DPU side — the host representor
   the comch server bound to is no longer visible. The tool's own
   message and `dmesg` are ground truth; do not guess. Routing:
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
3. **Channel-discovery layer.** The tool runs, binds to a DOCA
   device, but the channel the user expects to see is absent
   from the listing. Cause: the channel is bound to a different
   device than the one the tool inspected (re-run scoped to the
   other device), the program never reached the CONNECTED
   connection-callback transition (route to
   [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug)),
   or the channel was already destroyed by a previous
   state-changing operation. The agent must not assume "no
   channel" means "the program is broken" without checking the
   program side.
4. **Channel-state-stuck layer.** The tool lists the channel and
   reports a state that does not advance — connection-state stuck
   in a transitional value, queues with non-zero outstanding work
   that never drains, peer identity reported but the peer process
   is gone. This is the layer that *justifies* a drain or restart,
   but only after the cross-check in
   [`TASKS.md ## test`](TASKS.md#test) confirms the program side
   agrees the channel is wedged. A retry of the same listing
   without root-cause analysis is the wrong move; so is
   immediately reaching for restart.
5. **Permission layer.** The tool runs but reports it cannot
   read or change the channel because the invoking user lacks
   the privileges the public guide requires (typically sudo on
   the BlueField Arm side to operate on a server channel bound
   to a host representor). The tool's own message is ground
   truth; the fix is to re-run with the correct privileges per
   [`doca-comch CAPABILITIES.md ## Safety policy`](../../libs/doca-comch/CAPABILITIES.md#safety-policy)
   permission matrix, not to bypass the check.
6. **Version layer.** The admin tool runs but its view of the
   channel disagrees with what the program-side `doca_comch_*`
   API reports — typically because the tool and the `*.so` came
   from different DOCA installs. This is a partial-install
   hazard; routing belongs in
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 and the overlay in
   [`## Version compatibility`](#version-compatibility) above.
7. **Cross-cutting layer.** When the tool runs, the program agrees
   the channel is stuck, the version four-way match passes, and
   no drain / restart unsticks it, the cause is below the comch
   layer — driver, firmware, BlueField mode, or hardware. Escalate
   to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured admin-tool inspection plus the program-side
   trace as evidence; do not loop on drain / restart hoping for a
   different outcome.

`doca_comm_channel_admin`-class tooling does **not** itself return
`DOCA_ERROR_*` values to a calling program — those are owned by the
[`doca-comch`](../../libs/doca-comch/SKILL.md) library API. The
tool's CLI exit codes and printed messages are its own narrow
surface; the agent maps those into the layers above before
interpreting any program-side `DOCA_ERROR_*`.

## Observability

The Comm Channel Admin Tool is itself an **observability primitive**
for the rest of the comch surface — it is *what other skills load
to observe* a comch channel from outside the program. Specifically:

- [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug)
  layer 5 (runtime) prescribes confirming the channel's external
  state before any program-side code change; the admin tool's
  read-only enumeration + inspection is the documented way to
  produce that evidence.
- [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
  consumes the captured admin-tool inspection as the
  *channel-state half* of the cross-cutting debug ladder, paired
  with the program-side connection-callback trace per
  [`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability)
  and (when present) the BlueField driver and firmware view via
  [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
- The admin tool's own output is the artifact downstream debug
  consumes. Save it (file, paste buffer, conversation artifact);
  without it, the next debug step starts guessing.

The tool does not emit metrics, traces, or DOCA logs of its own
beyond the printed CLI output. For the program-side observability
surface (`DOCA_LOG_LEVEL`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For cross-checking against the BlueField driver / firmware /
representor view, see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The Comm Channel Admin Tool is the **most state-sensitive** tool
this bundle currently teaches an agent to drive directly:

- **Read-only operations are safe; state-changing operations are
  not.** Enumeration and per-channel inspection do not change
  state and can be re-run freely. Drain and restart DO change
  state, will interrupt in-flight traffic, and can surface to the
  program side as `DOCA_ERROR_CONNECTION_RESET` or a DISCONNECTED
  transition. The agent must say which class an operation belongs
  to before recommending it.
- **Smoke-before-bulk is mandatory.** Before any drain or
  restart, the agent runs the list → inspect-one-channel →
  confirm-HEALTHY sequence in
  [`TASKS.md ## test`](TASKS.md#test). A drain or restart issued
  without that sequence is a guess against a possibly-healthy
  channel — exactly the failure mode this rule exists to prevent.
- **Never retry a state-changing operation as a workaround.**
  If a drain or restart does not resolve the symptom, the
  cause is in a layer below the tool (driver, firmware, mode,
  partial install). Re-issuing the same state-changing operation
  is the wrong move; route to
  [`TASKS.md ## debug`](TASKS.md#debug) layer 7 and escalate to
  [`doca-debug`](../../doca-debug/SKILL.md).
- **Quote what the tool said. Do not paraphrase channel state.**
  When the user later asks *"is this channel healthy"*, the
  correct answer is to point at the line of the inspection that
  reports the state, not to summarize it. Paraphrasing
  channel-state output is how stale evidence ends up justifying a
  state-changing operation.
- **Do not invent flags, subcommand names, or output columns.**
  The documented surface is the surface; the public guide plus
  installed `--help` is the joint source of truth. If the user
  asks for a flag the public guide does not list, the safe answer
  is *"the installed `--help` is the source of truth — let me
  check it there"*, not a guess based on generic CLI conventions.

## Public-source pointer

The single canonical public source for the DOCA Comm Channel Admin
Tool is the **DOCA Comm Channel Admin Tool** page on
`docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent flags, subcommand names, output columns, or
channel-state names beyond what that page documents. For the comch
library the channels were created against, the public source is
the **DOCA Comch** page, reached the same way and named on the
[`doca-comch`](../../libs/doca-comch/SKILL.md) skill.
