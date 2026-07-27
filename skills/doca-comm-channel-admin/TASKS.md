# DOCA Comm Channel Admin Tool — Tasks

> **Actual binary contract.** Run
> `/opt/mellanox/doca/tools/doca_comm_channel_admin` with zero
> application arguments. It scans every comch-capable device on
> the current side through `resourcedump` and prints SERVERS and
> CONNECTIONS tables. No list, inspect, device-scope, drain,
> restart, or other application operation exists.

**Where to start:** The verbs that carry real workflow content are
`## run`, `## test`, and `## debug`. The other three (`configure`,
`build`, `modify`) are documented routing stubs that exist because
the bundle's verb contract is uniform. `## test` cross-checks the
single scan against the program side.

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
  zero-argument scan (see [`## run`](#run)), inspect the printed
  rows, and let the
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

The Comm Channel Admin Tool exposes one operation: a
zero-application-argument scan-and-print.

1. **Confirm the binary is present** under
   `/opt/mellanox/doca/tools/`. If absent, route to
   [`## configure`](#configure) above.
2. **Run the binary with zero application arguments** on the side
   where the user reports the problem. Do not add a device selector
   or a list/inspect token; the binary itself scans every
   comch-capable `doca_dev` on that side.
3. **Capture the process result as one artifact:** exit status,
   complete stdout (SERVERS and CONNECTIONS tables), and complete
   stderr.
4. **Check the `resourcedump` dependency before interpreting empty
   output.** Missing MFT, `resourcedump` absent from `PATH`,
   insufficient privilege, non-zero child exit, or child stderr is
   a setup failure. Route it to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); do not
   call it an empty inventory.
5. **Read the relevant row.** Locate the expected server or
   connection in the printed tables. There is no per-row follow-up
   invocation. If the row is absent after a clean exit, cross-check
   the program side and, if needed, run the same zero-argument scan
   on the other side.

When recording the run for downstream consumers, write down: the
DOCA version (per [`doca-version`](../../doca-version/SKILL.md)),
the side the tool was run on (host vs BlueField Arm), the exact
zero-argument command line, exit status, stdout tables, and stderr.
The downstream `## test` and `## debug` workflows depend on all
six fields.

## test

`## test` validates that one complete scan is trustworthy and
cross-checks its relevant row; it does not invoke a second
operation.

1. **Require a clean scan artifact.** Per [`## run`](#run), retain
   exit status, both stdout tables, and stderr. Any unresolved
   `resourcedump` prerequisite or execution failure routes to
   setup and blocks interpretation.
2. **Locate the expected row in the existing tables.** Do not add
   an inspect or device-scope argument. Quote the row verbatim.
3. **Cross-check against the program side.** The program-side
   connection callback in
   [`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability)
   must agree with the printed row.
4. **If the expected row is absent, scan the other side.** Run the
   same zero-argument binary there and compare its full tables.
   There is still no device-scoped or per-channel invocation.

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
2. **`resourcedump` prerequisite / execution layer.** Confirm MFT
   is installed, `resourcedump` resolves through `PATH`, and the
   documented privilege requirement is met. Capture the admin
   tool's exit status and stderr and the child `resourcedump`
   exit/stderr. Any non-zero exit or diagnostic output is routed
   to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
   do not interpret absent tables.
3. **Device-binding layer.** The scan cannot read one or more
   devices on this side. Use the captured exit/stderr and
   `resourcedump` diagnostics as ground truth and route driver,
   representor, privilege, MFT, or PATH remediation to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
4. **Row-discovery layer.** The scan exits cleanly and prints both
   tables, but the expected row is absent. Inspect those rows; do
   not rerun with an invented device scope. Confirm the
   program-side connection callback has fired CONNECTED per
   [`doca-comch TASKS.md ## debug`](../../libs/doca-comch/TASKS.md#debug),
   then run the same zero-argument scan on the other side if
   needed.
5. **Version layer.** Tool runs but its view of the channel
   disagrees with what the program-side `doca_comch_*` API
   reports. Walk the four-way match per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 and apply the admin-tool overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
   When the tool and the `*.so` came from different installs, the
   fix is a consistent reinstall, not a code change.
6. **Cross-cutting layer.** All layers above are clean but the
   scan and program-side evidence disagree. Escalate to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured admin-tool inspection plus the program-side
   trace as evidence.

In every case: quote the relevant table row and retain the exact
exit status and stderr. Do not synthesize a second command surface
from the printed fields.

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

The tool has exactly one application invocation class.

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
| Scan and print all comch rows on this side | `/opt/mellanox/doca/tools/doca_comm_channel_admin` with zero application arguments; capture exit status, complete stdout, and complete stderr | [`## run`](#run), [`## test`](#test), [`## debug`](#debug) | Exit 0; MFT `resourcedump` is installed and reachable through `PATH`, privilege is sufficient, no child invocation fails, and stdout contains the SERVERS and CONNECTIONS tables. Any MFT/PATH/privilege/non-zero-exit/stderr failure routes to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) before table interpretation. |

Three cross-cutting rules for this appendix:

- **Never add an application argument.** There is no subcommand,
  device scope, or per-row follow-up.
- **Inspect rows or the other side.** If the target row is absent,
  inspect the existing tables or run the same zero-argument scan
  on the other side.
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

- The binary has one zero-application-argument scan-and-print
  operation. Do not invent additional command surfaces.
- **Quote, do not paraphrase.** The table output is the
  artifact downstream debug consumes; reformatting it loses
  fidelity that the rest of the bundle's procedures depend on.
- This skill **assumes a healthy DOCA install** (or the public
  NGC DOCA container) at both endpoints of the host ↔ DPU pair.
  If the install is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) before running
  anything else here. For the comch programming surface that
  created the channel, see
  [`doca-comch`](../../libs/doca-comch/SKILL.md).
