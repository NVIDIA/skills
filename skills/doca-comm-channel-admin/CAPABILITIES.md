# DOCA Comm Channel Admin Tool — Capabilities

> **Actual binary contract.** The tool has one read-only operation:
> invoke it with zero application arguments, let it scan every
> comch-capable device on the current side through `resourcedump`
> (MFT), and read the SERVERS and CONNECTIONS tables. It has no
> per-device scope, per-row follow-up, or state-changing operation.

**Where to start:** The tool is a single admin CLI; the pattern
overview below names the recurring admin-tool questions. Pick the
pattern first, then drill into the H2 that owns the substance. For
the *how* of executing each pattern, jump to [TASKS.md](TASKS.md).
For the comch programming surface the channels were created
against, see
[`doca-comch CAPABILITIES.md`](../../libs/doca-comch/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents what
the tool reports, what versions it ships in, the layered error
and observability surfaces, and its read-only safety policy.

## Pattern overview

Every Comm Channel Admin Tool question this skill teaches resolves
into one of THREE patterns. The patterns are CLASSES — they apply
across every DOCA install that ships the tool, not just one
platform.

| Admin-tool pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Scan and print | Run the zero-application-argument binary once; it scans every comch-capable device on this side and prints SERVERS and CONNECTIONS tables | [`## Capabilities and modes`](#capabilities-and-modes) + [TASKS.md ## run](TASKS.md#run) |
| 2. Cross-check against the program | Match the relevant printed row with the program-side `doca_comch_*` connection callback; if needed, run the same zero-argument scan on the other side | [`## Observability`](#observability) + [TASKS.md ## test](TASKS.md#test) |
| 3. Diagnose empty / mismatched / failed output | Map absent rows or a `resourcedump` failure to the correct layer before any code change | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **One invocation only.** Do not synthesize a list, inspect, or
  device-selection argument. Locate the target in the printed rows
  or run the same zero-argument scan on the other side.
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
registers ZERO application-level arguments. It performs only the
scan-and-print operation described here.

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

Changing channel or device state is outside this binary and this
skill. Route program lifecycle work to
[`doca-comch`](../../libs/doca-comch/SKILL.md), and driver or
device work to [`doca-setup`](../../doca-setup/SKILL.md) plus
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).

The tool runs on **both sides of a host ↔ DPU pair** — the host
sees host-side clients (each bound to a BlueField PCIe address),
and the BlueField Arm sees the DPU-side server (bound to a host
representor). The set of channels visible from one side is not
necessarily symmetric with the other side; the cross-checking
pattern in [`TASKS.md ## test`](TASKS.md#test) is how the agent
reconciles the two views.

There is no application subcommand or device-scope argument to
discover. Use the zero-argument operation and read the table
headings emitted by the installed binary.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The Comm Channel Admin Tool-specific overlay** is:

- **The tool ships under the `Comm Channel` name even on DOCA installs ≥ 2.5.** The companion library was renamed to `doca-comch` in DOCA 2.5 (see [`doca-comch CAPABILITIES.md ## Version compatibility`](../../libs/doca-comch/CAPABILITIES.md#version-compatibility) for the rename); the admin tool's public guide URL slug remains `DOCA-Comm-Channel-Admin-Tool`. An agent searching for *"DOCA Comch Admin"* on `docs.nvidia.com` should fall back to the Comm Channel Admin Tool guide and surface the naming asymmetry to the user, not assume the tool is missing.
- **Confirm the tool is present before assuming availability.** If the user reports the binary is absent under `/opt/mellanox/doca/tools/`, the right answer is to confirm the installed DOCA version per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure) and route to [`doca-setup`](../../doca-setup/SKILL.md) for an upgrade or reinstall, not to recommend a wrapper script that simulates the tool from outside.
- **Where it runs:** on the x86 / Arm host that has DOCA installed, *or* on the BlueField Arm side. Same binary, same flags; the set of channels it sees differs by side per [`## Capabilities and modes`](#capabilities-and-modes).
- **Tool version must match the comch `*.so` it inspects.** A program built against one DOCA train inspected by an admin tool from a different train is the same partial-install hazard as case (a) ≠ (c) in the four-way-match rule. When in doubt, run `pkg-config --modversion doca-comch` and the tool's own `--version` per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) and quote both.

## Error taxonomy

The error layers the agent should distinguish, in escalating order:

1. **Tool-not-installed.** The admin-tool binary does not exist
   under `/opt/mellanox/doca/tools/`. Cause: DOCA is not installed
   on this host, the install does not include the Comm Channel
   tooling subpackage, or the install version pre-dates the
   tool's availability. Routing:
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and the version-compatibility overlay above.
2. **`resourcedump` prerequisite / execution layer.** The binary
   depends on the MFT `resourcedump` executable for each scanned
   device. If MFT is not installed, `resourcedump` is absent from
   `PATH`, privileges are insufficient, or a child invocation
   exits non-zero, capture the admin tool's exit status and full
   stderr plus the failing `resourcedump` exit/stderr. Do not
   interpret missing tables as “no channels.” Route MFT install,
   PATH, and privilege remediation to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug), then
   rerun the same zero-argument scan.
3. **Device-binding layer.** The tool runs but cannot scan a DOCA
   device on this side. The underlying driver stack may be absent
   or the expected representor invisible. The tool's exit status,
   stderr, and `resourcedump` diagnostics are ground truth.
   Routing: [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
4. **Row-discovery layer.** The scan exits successfully and prints
   both tables, but the expected server or connection row is
   absent. Inspect the rows already printed; do not rerun with an
   invented device scope. Cross-check the program callback and, if
   necessary, run the same zero-argument scan on the other side.
   The program may never have reached the CONNECTED
   transition (route to
   [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug)),
   or the channel may already have been destroyed.
5. **Version layer.** The admin tool runs but its view of the
   channel disagrees with what the program-side `doca_comch_*`
   API reports — typically because the tool and the `*.so` came
   from different DOCA installs. This is a partial-install
   hazard; routing belongs in
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 and the overlay in
   [`## Version compatibility`](#version-compatibility) above.
6. **Cross-cutting layer.** When the scan and program-side view
   disagree after the version match passes, the cause may be below
   the comch layer — driver, firmware, BlueField mode, or hardware. Escalate
   to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured admin-tool inspection plus the program-side
   trace as evidence.

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

The Comm Channel Admin Tool is read-only:

- **Invoke only the zero-application-argument scan.** Do not add
  list, inspect, device-scope, drain, restart, or other invented
  arguments.
- **Preserve evidence.** Capture exit status, stdout tables, and
  stderr together. Quote relevant SERVERS / CONNECTIONS rows
  verbatim.
- **Fail closed on `resourcedump`.** Missing MFT, a missing PATH
  entry, insufficient privilege, non-zero child exit, or child
  stderr is a setup failure, not an empty-channel result. Route to
  [`doca-setup`](../../doca-setup/SKILL.md) before interpreting
  the tables.

## Public-source pointer

The single canonical public source for the DOCA Comm Channel Admin
Tool is the **DOCA Comm Channel Admin Tool** page on
`docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent application arguments or output columns beyond what
that page and the installed binary document. For the comch
library the channels were created against, the public source is
the **DOCA Comch** page, reached the same way and named on the
[`doca-comch`](../../libs/doca-comch/SKILL.md) skill.
