---
license: Apache-2.0
name: doca-comm-channel-admin
description: >
  Use this skill to enumerate host↔DPU DOCA comch (formerly
  Comm Channel) servers and connections via the shipped
  doca_comm_channel_admin binary — listing comch-capable
  devices and decoding the per-device server / connection
  table (server name, PID, in-use / max, PCIe address). The
  shipped binary is a SINGLE-SHOT SCAN-AND-PRINT tool with no
  registered arguments — NO list / inspect / drain / restart
  subcommands; one inventory pass over every comch-capable
  doca_dev on this side. Channel reset / drain / restart go to
  doca-comch (program side), doca-setup / doca-hardware-safety
  (driver reload), or BFB / RShim — NOT to this binary.
  Trigger on phrasings like "list comch servers", "which
  channels are active on this BlueField", or "verify admin
  tool sees same channel as program." Refuse and route
  elsewhere for the comch programming API, library install,
  protocol design, channel reset, or general orientation.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with the Comm Channel tooling
  subpackage so the CLI is present under
  /opt/mellanox/doca/tools/. Runs from either the x86/Arm host or
  the BlueField Arm side; a live host↔DPU comch channel created
  via the doca-comch library is needed for non-empty inspection.
---

# DOCA Comm Channel Admin Tool

> **CRITICAL — how to read the body of this skill (drain / restart /
> inspect / state-changing-operation framing).** The shipped
> `doca_comm_channel_admin` binary is a SINGLE-SHOT, READ-ONLY,
> SCAN-AND-PRINT tool. It registers ZERO application-level arguments
> in `register_comm_channel_admin_params()` (only ARGP defaults are
> registered), it shells out to `resourcedump` (MFT) per device, and
> it prints two ASCII tables. There is NO `drain` subcommand, NO
> `restart` subcommand, NO `inspect <channel-id>` subcommand, NO
> state-changing operation, and NO `list-first → inspect-one →
> drain-or-restart` workflow exposed BY THIS BINARY. Wherever the
> sections below (`## Example questions`, [`CAPABILITIES.md`](CAPABILITIES.md),
> [`TASKS.md`](TASKS.md)) say *"drain a channel"*, *"restart it"*,
> *"per-channel inspect step"*, *"state-changing operation gate"*,
> or *"smoke-before-bulk"*, read those as **conceptual operator
> actions the operator performs via OTHER paths AFTER reading this
> tool's tables** — not as subcommands the admin binary itself ships.
> Concretely: the operator drains/resets/reconnects by going to the
> **program side** through [`doca-comch`](../../libs/doca-comch/SKILL.md)
> (the comch library's `doca_ctx_stop()` → reset → `doca_ctx_start()`
> reconnect lifecycle), or by reloading the BlueField driver / mode
> via [`doca-setup`](../../doca-setup/SKILL.md) +
> [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md), or by
> RShim / BFB reset for the deepest cases — NEVER by re-invoking the
> admin binary with a flag, because no such flag exists. The body
> below uses the legacy *drain/restart/inspect/state-changing* names
> because they describe the operator's mental model; this banner is
> the authoritative mapping from that model onto the actual binary
> surface.

**Where to start:** This is a tool skill for invoking the DOCA Comm
Channel Admin Tool — the **read-only inventory** CLI counterpart
to the [`doca-comch`](../../libs/doca-comch/SKILL.md) library. The
shipped `doca_comm_channel_admin` binary takes **no arguments
beyond ARGP defaults** (`--help`, `--version`, `--log-level`,
`--sdk-log-level`, `--json`) and performs **one inventory pass**
per invocation: it walks every `doca_dev` on this side, filters
to comch-capable devices, shells out to `resourcedump` (MFT) on
each, and prints two ASCII tables (SERVERS and CONNECTIONS).
There is no `list` subcommand, no `inspect` subcommand, no
`drain` flag, and no `restart` flag — those are not part of the
tool's surface. Open [`TASKS.md`](TASKS.md) and start at
[`## run`](TASKS.md#run) for the single-shot invocation, or
[`## debug`](TASKS.md#debug) when the user reports the tool sees
a different channel set than the program. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
the printed tables actually mean* and *what is not in this
tool's scope*. If the user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user
needs MFT (`mst start` + `resourcedump` on PATH and root/sudo
privileges to invoke it), `doca-setup` + `doca-public-knowledge-map`
cover that. If the user is holding pre-2.5 docs that mention
"Comm Channel", route to
[`doca-comch CAPABILITIES.md ## Version compatibility`](../../libs/doca-comch/CAPABILITIES.md#version-compatibility)
for the rename rule. If the user wants to *change* channel
state (reset / drain / restart), route to the program-side
reconnect lifecycle in [`doca-comch`](../../libs/doca-comch/SKILL.md)
or to BlueField mode / driver reload in
[`doca-setup`](../../doca-setup/SKILL.md) +
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — **not**
to this tool.

## Example questions this skill answers well

The CLASSES of admin-tool questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece;
the worked example is one instance.

- **"Which comch channels are currently active on this BlueField?"** —
  worked example: *"enumerate every comch channel the DPU server is
  presently accepting from host clients"*. Answered by the channel
  enumeration in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the list-first invocation in
  [`TASKS.md ## run`](TASKS.md#run).
- **"What state is this channel in?"** — worked example: *"a host
  client reports its send-task is hanging; is the channel HEALTHY,
  draining, or stuck"*. Answered by the channel-state inspection
  surface in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the per-channel inspect step in
  [`TASKS.md ## run`](TASKS.md#run).
- **"A channel is stuck — should I drain it or restart it?"** —
  worked example: *"a producer / consumer fast-path has no
  completions for 30 seconds and the consumer process is still
  alive"*. Answered by the drain-vs-restart decision rule in
  [`TASKS.md ## debug`](TASKS.md#debug) +
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  state-changing-operation gate.
- **"How do I know the admin tool's view matches what my Comch
  program sees?"** — worked example: *"the program reports
  CONNECTED but the admin tool lists zero channels"*. Answered by
  the cross-checking pattern in
  [`TASKS.md ## test`](TASKS.md#test) and the representor-binding
  layer in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- **"Is this admin tool on my installed DOCA version, and does it
  match the comch library version?"** — worked example: *"is the
  tool available on DOCA 2.4"*. Answered by the overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) rules and adds the
  Comm Channel Admin Tool specifics.
- **"The tool prints nothing — is the install broken or is there
  genuinely no channel?"** — worked example: *"`list` returned an
  empty result on a host with a known-good Comch client"*.
  Answered by the empty-output interpretation rules in
  [`TASKS.md ## debug`](TASKS.md#debug) +
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

## Audience

This skill serves **external operators and AI agents who need to
inspect or intervene on a host-DPU comch channel from the outside**
— after the [`doca-comch`](../../libs/doca-comch/SKILL.md) library
has been used to create the channel from a program. Concretely:

- A platform operator who runs a Comch-using service on BlueField
  and needs to confirm the channel is healthy before declaring the
  service ready.
- A developer of a Comch consumer who sees `DOCA_ERROR_AGAIN` or a
  silent stall on the program side and wants to read the channel's
  state from outside the program rather than guessing.
- An AI agent driving an operational triage step *"is the comch
  channel stuck"* before recommending a code change or a process
  restart.

It is **not** for users debugging the admin tool itself, **not** a
substitute for the live public DOCA Comm Channel Admin Tool guide,
and **not** the right place for users learning the comch API —
that audience belongs in
[`doca-comch`](../../libs/doca-comch/SKILL.md).

The tool is shipped as a CLI binary under
`/opt/mellanox/doca/tools/`, not a library you link against. The
skill uses the same `kind: tool` three-file shape as the rest of
the bundle so the agent's task-verb contract
(`configure / build / modify / run / test / debug`) is uniform
across libraries, services, and tools.

## When to load this skill

Load this skill when the user is — or the agent needs to — invoke
the DOCA Comm Channel Admin Tool on a real host or BlueField Arm
with DOCA installed (or inside the public NGC DOCA container with
the right device passthrough). Concretely:

- Listing currently active comch channels on a host or DPU.
- Inspecting the state of one named channel before deciding
  whether to leave it alone, drain it, or restart it.
- Walking the drain-vs-restart decision for a stuck channel
  reported by a Comch-using application.
- Cross-checking the admin tool's view against the program-side
  connection callback state when the two appear to disagree.
- Capturing a side-effect-free channel snapshot as prerequisite
  evidence for a later debug session that crosses program /
  channel / driver layers.

Do **not** load this skill for general DOCA orientation, the comch
programming API, library install, or comch protocol design. For
those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-comch`](../../libs/doca-comch/SKILL.md), or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what the Comm Channel Admin Tool reports and
  changes: the documented channel-state surface, the read-only vs
  state-changing operation split, the version-availability overlay
  that redirects to [`doca-version`](../../doca-version/SKILL.md),
  the layered error taxonomy (tool-not-installed / device-binding
  / channel-discovery / channel-state-stuck / permission /
  version / cross-cutting), the tool's role as an observability
  primitive for [`doca-comch`](../../libs/doca-comch/SKILL.md)
  debug sessions, and the safety policy that makes drain and
  restart high-stakes operations gated on a clean inspection
  first.
- `TASKS.md` — step-by-step workflows for the in-scope task verbs:
  `configure` (route to install), `build` (route to install),
  `modify` (refuse), `run` (list-then-inspect), `test`
  (smoke-before-bulk capability check), `debug` (the layered
  diagnosis ladder), plus a `Deferred task verbs` block and a
  `Command appendix` that honors the bundle's
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  preamble.

The skill assumes a host or BlueField where DOCA is already
installed (or the public NGC DOCA container is running with the
right device passthrough) and the operator has whatever privileges
the public DOCA Comm Channel Admin Tool guide requires.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle.
To keep the boundary clean, it deliberately does not contain — and
pull requests should not add:

- **Verbatim flag inventories, subcommand names, or output column
  names.** The public DOCA Comm Channel Admin Tool guide on
  `docs.nvidia.com` and the installed `--help` on the user's
  version are the joint source of truth; copying them here pins
  the skill to one release and silently rots when the tool
  evolves. The skill routes the agent at those sources instead.
- **Pre-baked example output.** Output is install-, version-, and
  channel-state-specific. A captured example will mislead an
  operator on a different platform / state.
- **Wrappers, parsers, or scripts** in any language that consume
  the admin tool's output. The output format is documented; users
  who want to script against it should read the live guide and
  write the parser against their installed version.
- **A `samples/` or `reference/` subtree.** This is a thin loader
  for a documented CLI; substantive material lives on the public
  page and in `--help`.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (the user wants to inspect or intervene on a comch
   channel from the outside, not learn the comch API).
2. **For what the tool reports, the read-only vs state-changing
   split, version availability, the layered error surface,
   observability, and safety posture, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the smoke-before-bulk
   workflow — `configure`, `build`, `modify`, `run`, `test`,
   `debug`, plus the `Command appendix` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-comch`](../../libs/doca-comch/SKILL.md) — the library
  whose channels this tool inspects. Pair them in every
  triage session: the program-side connection callback and the
  admin tool's channel state are the two halves of the same
  picture.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Comm Channel Admin Tool guide and
  the rest of the public DOCA documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. The `## Version compatibility` section
  in [`CAPABILITIES.md`](CAPABILITIES.md) is a concise overlay
  that redirects here for the body.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back → report contract for
  structured helper tools. The Command appendix in
  [`TASKS.md`](TASKS.md) honors this contract.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, representor visibility checks, and the
  *I have no install yet* path with the public NGC DOCA
  container. This skill assumes its preconditions are satisfied.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. The Comm Channel Admin Tool slots in at the
  *runtime* layer as the read-only inspection surface before any
  code change is recommended.
