# DOCA Socket Relay — Capabilities

**Where to start:** The Socket Relay is a single DOCA artifact that
can be deployed in more than one shape; the pattern overview below
names the recurring relay-class questions. Pick the pattern first,
then drill into the H2 that owns the substance. For the *how* of
executing each pattern, jump to [TASKS.md](TASKS.md). For the
host ↔ DPU control-plane sibling whose channels coordinate the
endpoints the relay carries traffic between, see
[`doca-comch CAPABILITIES.md`](../../libs/doca-comch/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
the relay carries and changes*, *which deployment shapes are
documented*, *what versions it ships in*, *the layered error and
observability surfaces*, and *the high-stakes safety policy that
gates the data-path pieces*. For step-by-step invocations and the
smoke-before-bulk workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every Socket Relay question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
DOCA install that ships the relay, not just one platform.

| Relay pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Migrate a socket-based app onto the DOCA fabric | Use the relay to terminate the application's host-side socket locally and forward the bytes across the host ↔ DPU boundary, without rewriting the application against the DOCA programming surface | [`## Capabilities and modes`](#capabilities-and-modes) use-case framing + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Pick the deployment shape | Choose how the relay runs on the host ↔ DPU pair (in-process beside the app vs sidecar process / container vs service container on the BlueField) for the operator's environment | [`## Capabilities and modes`](#capabilities-and-modes) deployment-shape axis + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 3. Configure the socket and the forwarding endpoint | Point the relay at the host application's AF_UNIX (UDS) socket via `-s/--socket <path>` and select the DPU-side Comch service name via `-n/--cc-name` plus the DOCA device PCIe address (`-p/--pci-addr`, plus `-r/--rep-pci` on the DPU side). The shipped binary uses an AF_UNIX socket on the host side — no TCP / UDP framing; the relay's network leg is Comch, not raw IP. | [`## Capabilities and modes`](#capabilities-and-modes) socket-type and forwarding-endpoint axes + [TASKS.md ## configure](TASKS.md#configure) steps 3-4 |
| 4. Smoke-before-bulk admit a fleet | Bind one relay endpoint, confirm one host application client connects, confirm one round-trip succeeds end-to-end, only then admit the rest of the fleet | [TASKS.md ## test](TASKS.md#test) eval loop + [`## Safety policy`](#safety-policy) data-path posture |
| 5. Diagnose a stuck or silent relay | Map the symptom (host app cannot connect, bytes leave the host but never arrive, fleet works in test but breaks in production) to the right layer of the error taxonomy before any state-changing intervention | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Pair the relay with its control-plane sibling | Use [`doca-comch`](../../libs/doca-comch/SKILL.md) to coordinate the relay's endpoints and lifecycle when the application owner needs programmatic control over which DPU-side terminator a host client is bridged to | [`## Observability`](#observability) cross-cutting pairing + [`SKILL.md ## Related skills`](SKILL.md#related-skills) |

Two cross-cutting rules that apply to *every* pattern above:

- **The relay sits in the data path.** A misconfigured forwarding
  endpoint, a stale relay bound on the wrong socket, or a teardown
  that races with the host application produces a *silent*
  break — the application observes connect / send / recv errors
  whose root cause is the relay, not the application. The agent
  must say so before any state-changing operation, and must
  recommend the no-relay control comparison documented in
  [`## Safety policy`](#safety-policy) before promoting a relay
  deployment to production.
- **The Socket Relay is one half of the host ↔ DPU bridging
  picture; comch is the other half.** The relay carries
  socket-shaped bytes across the boundary; comch carries
  programmatic control messages across the same boundary. An
  agent that recommends the relay without naming comch (when the
  operator needs programmatic control of the bridge), or vice
  versa, is missing half the design space documented in
  [`SKILL.md ## Related skills`](SKILL.md#related-skills).

## Capabilities and modes

The DOCA Socket Relay is a documented DOCA artifact that ships on
installs which include it. The exact installed location, binary
name, packaging shape, and per-platform availability live in the
public DOCA Socket Relay guide reachable via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
and on the installed `--help`; the skill deliberately does not
pin them — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

**Use case — what the relay is *for*.** The Socket Relay is the
documented bridge for **migrating an existing socket-based
application onto the DOCA fabric without rewriting it**. The host
application keeps speaking its socket protocol locally, exactly
as it does today; the relay terminates that socket on the host
side, forwards the byte stream / datagram across the host ↔ DPU
boundary, and a DPU-side terminator (the relay's far half, or a
documented service container on the BlueField) re-presents the
traffic to the DPU-side peer. From the application's point of
view, nothing changed except *where* its peer lives.

**Three-axis configuration model — the load-bearing concept.**
Every Socket Relay deployment commits to a point in this space;
omitting any axis produces an ambiguous answer the agent must
flag back to the user.

| Axis | What it picks | Why the agent must name it |
| --- | --- | --- |
| 1. Deployment shape | How and where the relay process runs — *in-process* (linked or co-resident next to the host application), *sidecar* (a separate process / container next to the application on the same host), or *container* (a service container on the BlueField via the documented kubelet-standalone runtime). | Each shape has a different precondition surface (host-OS permissions for in-process; lifecycle coupling for sidecar; image-pull + static-pod + per-service config mount per [`doca-container-deployment CAPABILITIES.md ## Capabilities and modes`](../../doca-container-deployment/CAPABILITIES.md#capabilities-and-modes) for the container shape). An answer that picks one shape without naming the alternatives has silently narrowed the relay to a single use case. |
| 2. Socket type / protocol on the host leg | The shipped `doca_socket_relay` binary uses AF_UNIX (Unix Domain Sockets, SOCK_STREAM) on its host leg — the listener and acceptor threads in the shipped binary are all AF_UNIX, and the relay's `-s/--socket` argument is a filesystem path, not an IP/port pair. The DPU-network leg is the DOCA Comch transport (`-n/--cc-name` selects the named Comch service). If the public DOCA Socket Relay guide on the user's installed version adds TCP / UDP / etc. variants in a later release, the agent verifies that on the user's `--help` output before quoting any non-AF_UNIX behavior — it does not invent transport modes the shipped binary does not support. | Treating the relay as protocol-agnostic and quoting TCP / UDP debug rules against a UDS-only binary is a category error: framing, error semantics, file-descriptor permissions, and peer-credential propagation are AF_UNIX-shaped, not socket-API-generic. |
| 3. Forwarding endpoint | Where on the DPU side the relay forwards traffic to — the DPU-side terminator (the relay's far half, a DPU-side service, or a documented service container on the BlueField that re-presents the bytes to the DPU peer). | This is the most consequential axis for safety: a *correct* host-side bind with a *wrong* forwarding endpoint produces a relay that the host application can connect to but whose bytes go to the wrong place (or nowhere). The agent's diagnosis ladder in [`## Error taxonomy`](#error-taxonomy) treats forwarding-endpoint misconfiguration as its own layer for that reason. |

The three axes are **independent**; an in-process AF_UNIX relay
forwarding to one DPU-side terminator and a service-container
AF_UNIX relay forwarding to a different DPU-side terminator are
two separate deployments and the agent must keep them separate
when diagnosing. (Both legs are AF_UNIX-on-host + Comch-on-DPU
per the shipped binary; the host leg is never TCP / UDP.)

**Read-only operations vs state-changing operations.** Every
relay operation falls into one of two functional families,
exactly as the comm-channel-admin tool partitions its surface
(see
[`doca-comm-channel-admin CAPABILITIES.md ## Capabilities and modes`](../doca-comm-channel-admin/CAPABILITIES.md#capabilities-and-modes)
for the same shape):

- **Read-only operations.** Inspect the relay's running state —
  whether it is bound, on which side, against which socket /
  port / path, against which forwarding endpoint, with what
  current connection / session set. These cost nothing to run,
  do not interrupt traffic, and are the agent's default reach
  when the user reports a relay-side issue. They are the
  evidence base the rest of the workflow consumes.
- **State-changing operations.** Bind a new relay endpoint,
  repoint a forwarding endpoint, drain or tear down a running
  relay, restart a relay container. These DO change relay state
  and **DO sit in the data path** — interrupting them while host
  applications are connected can surface to the application as
  socket disconnects, send / recv errors, or partial-byte
  corruption in the cross-version boundary. They are reserved
  for the diagnosis ladder in [`TASKS.md ## debug`](TASKS.md#debug)
  after a clean read-only inspection has identified the cause.

**Where the relay can run.** Per the public guide, the relay can
run on the **host** side, on the **BlueField Arm** side, or on
*both* sides of a host ↔ DPU pair (one half terminates the host
socket; the other half re-presents the bytes to the DPU peer).
The deployment-shape axis above picks which side(s) the operator
runs the relay on for a given application. The set of running
relays visible from one side is not necessarily symmetric with
the other side; the cross-checking pattern in
[`TASKS.md ## test`](TASKS.md#test) is how the agent reconciles
the two views.

The exact, current binary name, command syntax, supported socket
types, supported forwarding-endpoint shapes, and output column
names live in the public guide and in `--help` on the installed
version. The skill deliberately does not pin them.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The Socket Relay-specific overlay** is:

- **Confirm the relay is shipped on the user's installed DOCA before assuming availability.** The relay is documented as a DOCA artifact under
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools); whether the binary or container image is actually present on a given host depends on the install profile and the DOCA version. If the user reports the artifact is absent, the right answer is to confirm the installed DOCA version per
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
  and route to [`doca-setup`](../../doca-setup/SKILL.md) for an upgrade or reinstall, not to recommend a wrapper script that simulates the relay from outside.
- **Where it runs:** on the x86 / Arm host that has DOCA installed, or on the BlueField Arm side, or on both sides of a host ↔ DPU pair depending on the deployment shape. The exact same artifact can be invoked as a CLI process or, when shipped as a service container, deployed via the runtime contract in
  [`doca-container-deployment CAPABILITIES.md ## Version compatibility`](../../doca-container-deployment/CAPABILITIES.md#version-compatibility) (which adds a third version anchor — the per-service container tag — on top of host DOCA and BFB).
- **Relay version must match the DOCA fabric layer it sits on.** The relay is a DOCA artifact and consumes the same `doca-common` runtime as every other DOCA library; a relay binary from one DOCA train invoked against a different-train install is the same partial-install hazard as case (a) ≠ (c) in the four-way-match rule. When in doubt, run `pkg-config --modversion doca-common` and the relay's own `--version` per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) and quote both.
- **Public guide URL slug is `DOCA-Socket-Relay`.** Search results that point at older slugs or at relay material on third-party sites are not authoritative for the user's installed version; the canonical guide is reached via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools), not from agent-memory recall.

## Error taxonomy

The relay's error surface is **broader** than a read-only admin
tool because the relay both *carries* application bytes and *can
fail in the data path*. The error layers the agent should
distinguish, in escalating order:

1. **Tool-not-installed.** The Socket Relay artifact is not
   present on the host or BlueField the operator is invoking it
   from — no relay binary, no relay container image, no relay
   subpackage. Cause: DOCA is not installed on this side, the
   install does not include the Socket Relay subpackage, or the
   install version pre-dates the relay's availability. Routing:
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and the version-compatibility overlay above.
2. **Relay-not-bound (port / socket).** The relay process / container
   starts but cannot bind the host-side socket / port / UDS path
   the operator configured. Cause: the address is in use by
   another process, the configured path is on a filesystem the
   relay user cannot write to, the configured port is below 1024
   without the privileges the public guide requires, or the
   configured socket family is not supported by the installed
   version. The relay's own message is ground truth; do not guess.
   Routing: re-quote the configuration from the public guide and
   `--help` on the installed binary; route filesystem / privilege
   issues to the host-OS team and to
   [`doca-setup CAPABILITIES.md ## Safety policy`](../../doca-setup/CAPABILITIES.md#safety-policy).
3. **Host-app-not-connecting.** The relay is bound on its
   configured socket, but the host application reports it cannot
   reach the relay (`ECONNREFUSED`, connect timeout, write to a
   UDS path that does not exist from the application's
   perspective). Cause: the application is configured to connect
   to a different socket / port / path than the relay actually
   bound, the application is in a different network namespace /
   container / mount namespace than the relay, or a host
   firewall rule is dropping the connect. The agent must
   reconcile the application-side configuration with the
   relay-side bound endpoint *before* declaring either
   *"the relay is broken"* or *"the application is broken"*.
4. **DPU-side-terminator-not-reachable.** The host application
   connects to the relay successfully, the relay accepts the
   connection, **and yet bytes never arrive at the DPU-side
   peer**. This is the *silent* data-path failure mode and the
   reason the relay is high-stakes. Cause: the forwarding
   endpoint the relay is configured to point at is wrong (the
   DPU-side terminator is not running, is bound to a different
   address, has been torn down, or was never the right
   destination), the BlueField mode is incompatible with the
   relay's transport, or — when the relay's far half is shipped
   as a service container — the BlueField pod is not in
   `Running` per
   [`doca-container-deployment CAPABILITIES.md ## Error taxonomy`](../../doca-container-deployment/CAPABILITIES.md#error-taxonomy).
   The agent's diagnosis discipline: capture the relay's view
   (was the byte received from the host?), the BlueField-side
   terminator's view (was the byte delivered?), and the DPU-side
   peer application's view (was the byte processed?) before
   blaming any single layer. A retry of *"connect again"*
   without root-cause analysis is the wrong move.
5. **Permission layer.** The relay runs and accepts the
   configured options but reports it cannot bind the socket,
   open the UDS path, or join the network namespace it needs
   because the invoking user lacks the privileges the public
   guide names (typically sudo on the BlueField Arm side to
   open a socket that bridges to the host, or specific group
   membership on the host side for low-numbered ports / shared
   UDS paths). The relay's own message is ground truth; the
   fix is to re-run with the correct privileges per the public
   guide and the host-OS rules, not to bypass the check.
6. **Version layer.** The relay runs but its behavior disagrees
   with what the public guide describes — typically because the
   relay artifact and the underlying `doca-common` came from
   different DOCA installs (partial install), or because the
   relay container tag does not match the host DOCA / BFB
   versions on the BlueField (per the three-anchor rule in
   [`doca-container-deployment CAPABILITIES.md ## Version compatibility`](../../doca-container-deployment/CAPABILITIES.md#version-compatibility)).
   Routing: walk the four-way match per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 and apply the overlay in
   [`## Version compatibility`](#version-compatibility) above.
7. **Cross-cutting layer.** The relay is bound, the host
   application connects, the forwarding endpoint is verified
   reachable, the version four-way match passes — and the
   symptom remains. The cause is below the relay: kernel /
   driver, BlueField mode, firmware, host networking
   misconfiguration, or hardware. Escalate to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) with the
   captured relay-side inspection plus the host-side and
   DPU-side application traces as evidence; do not loop on
   bind / unbind / restart hoping for a different outcome.

The Socket Relay does **not** itself return `DOCA_ERROR_*` values
to a calling program — those are owned by the DOCA libraries
([`doca-comch`](../../libs/doca-comch/SKILL.md),
[`doca-eth`](../../libs/doca-eth/SKILL.md), and friends). The
relay's CLI / container exit codes and printed messages are its
own narrow surface; the agent maps those into the layers above
before interpreting any program-side error from a paired DOCA
library.

## Observability

The Socket Relay's observability surface is **its own runtime
state plus the application sides on either end of the bridge**.
There is no single relay metric system; the visibility comes from
the relay's printed / logged output, the application-side
behavior, and the DPU-side terminator's behavior, read together.

Three primary signals the agent should reach for:

- **Relay-side state.** The relay binary's observable surface — the
  `doca_socket_relay` binary registers four operational params
  (`-s/--socket`, `-n/--cc-name`, `-p/--pci-addr`, `-r/--rep-pci`)
  via the standard DOCA argp surface (`--help`, `--json-config`,
  `--sdk-log-level`, `--log-level` are inherited from DOCA argp),
  and its visibility into "what is the relay actually doing" is
  delivered via **DOCA logger output** (`DOCA_LOG_REGISTER`-driven
  category `SOCKET_RELAY`) controlled by `--sdk-log-level` and
  `DOCA_LOG_LEVEL`, NOT via a dedicated `--list-channels` /
  `--list-connections` flag — those flags do NOT exist in the
  shipped binary as of DOCA 3.3. For the container deployment shape,
  ENTRYPOINT log per
  [`doca-container-deployment CAPABILITIES.md ## Observability`](../../doca-container-deployment/CAPABILITIES.md#observability)
  carries the same logger stream. The agent's discipline is: confirm
  what the binary actually exposes via `--help` on the user's
  installed version before quoting any inspection flag, and prefer
  the logger stream as the always-available channel.
- **Host-application-side observability.** The host application's
  own connect / send / recv error reporting (`errno`, log lines,
  `ss` / `netstat` output for the socket the application opened,
  application-level timeouts). The agent treats this as the
  *demand-side* evidence — what the consumer of the relay
  observed, independent of what the relay reports.
- **DPU-side terminator observability.** Whatever logging /
  metrics surface the DPU-side peer (the relay's far half or the
  DPU-side service container) exposes — *did the byte arrive,
  was it processed, what response was emitted*. Without this
  side, a *"the relay accepted the connection but bytes never
  arrive"* report is half-grounded; the agent must say so.

The relay is itself an **observability primitive** for the
host ↔ DPU data plane the same way the comm-channel-admin tool
is for the host ↔ DPU control plane (see
[`doca-comm-channel-admin CAPABILITIES.md ## Observability`](../doca-comm-channel-admin/CAPABILITIES.md#observability)).
A captured relay inspection is the artifact downstream debug
consumes; without it, the next debug step starts guessing.

For the cross-library env-side observability primitives
(representor enumeration, port state, BlueField mode, firmware)
see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For program-side observability of the paired DOCA libraries (the
control-plane comch channel coordinating the bridge, or the
underlying packet I/O layer) see
[`doca-comch CAPABILITIES.md ## Observability`](../../libs/doca-comch/CAPABILITIES.md#observability)
and
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The Socket Relay is the **most data-path-sensitive** tool this
bundle currently teaches an agent to drive directly: it sits in
the byte path between a host application and its peer, and a
misconfiguration produces a *silent* break the application
observes as a generic socket error.

- **Read-only operations are safe; state-changing operations are
  not.** Inspecting the relay's bind / forwarding-endpoint /
  connection state does not change anything and can be re-run
  freely. Binding a new endpoint, repointing a forwarding
  endpoint, draining or tearing down a running relay, restarting
  a relay container — all interrupt traffic and may surface to
  the host application as socket disconnects, partial-byte
  corruption in the cross-version boundary, or hard
  `ECONNRESET`. The agent must say which class an operation
  belongs to before recommending it.
- **A misconfigured forwarding endpoint silently breaks app
  connectivity.** This is the canonical failure mode for any
  data-path bridge: the host application connects to the relay,
  the relay accepts, the host application *thinks it is
  talking to its peer*, and yet the bytes go nowhere (or, worse,
  somewhere the operator did not intend). The agent must surface
  this risk before recommending any forwarding-endpoint change,
  and must walk the diagnosis ladder in
  [`## Error taxonomy`](#error-taxonomy) layer 4 explicitly when
  the symptom fits the silent-break shape.
- **Run the no-relay control comparison before promoting to
  production.** Before admitting a fleet of host applications
  onto the relay, the agent should recommend a control comparison
  in which the same host application talks to the same DPU-side
  peer **without** the relay (e.g. via the existing socket path
  the application uses today, or via a direct connection to the
  DPU-side terminator if the deployment supports it). A relay
  that *only* works when the relay is in the picture, but does
  not reproduce the application's pre-relay behavior, is hiding
  a regression that the smoke-before-bulk loop cannot expose
  alone.
- **Smoke-before-bulk is mandatory.** Before any state-changing
  operation on a running relay, and before admitting more than
  one host application client onto a relay endpoint, the agent
  runs the bind → connect → round-trip sequence in
  [`TASKS.md ## test`](TASKS.md#test). A state-changing operation
  issued without that sequence is a guess against a possibly-OK
  relay — exactly the failure mode this rule exists to prevent.
- **Never retry a state-changing operation as a workaround.** If
  a bind, repoint, drain, or restart does not resolve the
  symptom, the cause is in a layer below the relay (host
  network, BlueField mode, partial install, kernel / driver,
  hardware). Re-issuing the same state-changing operation is
  the wrong move; route to
  [`TASKS.md ## debug`](TASKS.md#debug) layer 7 and escalate to
  [`doca-debug`](../../doca-debug/SKILL.md).
- **Quote what the relay said. Do not paraphrase relay state.**
  When the user later asks *"is this relay healthy"*, the
  correct answer is to point at the line of the inspection that
  reports the bind / forwarding-endpoint / connection state, not
  to summarize it. Paraphrasing relay-state output is how stale
  evidence ends up justifying a state-changing operation.
- **Do not invent binary names, flag strings, socket-path
  defaults, or port numbers.** The documented surface is the
  surface; the public DOCA Socket Relay guide plus installed
  `--help` is the joint source of truth. If the user asks for a
  flag the public guide does not list, the safe answer is *"the
  installed `--help` is the source of truth — let me check it
  there"*, not a guess based on generic CLI conventions.

## Public-source pointer

The single canonical public source for the DOCA Socket Relay is
the **DOCA Socket Relay** page on `docs.nvidia.com`, reachable
through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent binary names, flags, subcommand names, output
columns, socket-path defaults, or port numbers beyond what that
page documents. For the comch library that owns the host ↔ DPU
control-plane sibling pattern, the public source is the **DOCA
Comch** page, reached the same way and named on the
[`doca-comch`](../../libs/doca-comch/SKILL.md) skill. For the
service-container runtime when the relay is shipped as a
container, the public source is the **DOCA Container Deployment
Guide**, reached the same way and named on the
[`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
skill.
