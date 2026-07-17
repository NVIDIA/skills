---
license: Apache-2.0
name: doca-socket-relay
description: >
  Use this skill when the operator is driving the DOCA Socket Relay
  to bridge a socket-oriented host application onto a BlueField DPU
  peer without rewriting it — picking the deployment shape
  (in-process, sidecar, or BlueField service container), configuring
  the host-side socket and the DPU-side forwarding endpoint, walking
  the bind → connect → round-trip → admit-fleet smoke, or diagnosing
  a stuck/silent relay. Trigger even when the user does not
  explicitly mention "DOCA Socket Relay" — typical implicit phrasings
  include "move my socket app onto the BlueField without rewriting
  it", "host app gets ECONNREFUSED on the relay", "relay accepts the
  connection but bytes never arrive on the DPU side", "first
  round-trip works, the rest hang", "bridge an AF_UNIX (UDS) socket
  to a DPU peer over Comch", or "I want a sidecar that forwards my
  socket to the BlueField". Refuse and route elsewhere for the comch programming
  API, line-rate raw packet I/O via doca-eth, and DOCA
  install/bring-up — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU attached, plus the
  socket-oriented host application that will be migrated onto the
  fabric. Container-shape deployment additionally relies on the
  BlueField kubelet-standalone runtime per doca-container-deployment.
---

# DOCA Socket Relay

**Where to start:** This is a tool skill for invoking the DOCA Socket
Relay — the host ↔ BlueField bridge that lets a socket-oriented host
application terminate its sockets locally while the relay forwards
the traffic to a DPU-side terminator across the DOCA fabric. Open
[`TASKS.md`](TASKS.md) and start at [`## configure`](TASKS.md#configure)
for the deployment-shape × socket-type × forwarding-endpoint
decision, then [`## run`](TASKS.md#run) for the bind → connect →
round-trip flow. Open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what state can the relay carry, what does it report,
and what does its data-path posture imply*. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking about the host ↔ DPU **control plane** rather than the
**data plane** the relay carries, route to
[`doca-comch`](../../libs/doca-comch/SKILL.md) — the relay is the
data-plane counterpart to comch.

## Example questions this skill answers well

The CLASSES of socket-relay questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece;
the worked example is one instance.

- **"Can I move my existing socket-based application onto a DOCA
  fabric without rewriting it?"** — worked example: *"a host service
  speaks a socket protocol to a peer; I want the peer to live on
  the BlueField DPU instead, but I do not want to port the
  application to the comch programming surface"*. Answered by the
  use-case framing in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the deployment-shape decision in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Where does the relay sit, and what runs on the host vs the
  DPU?"** — worked example: *"do I run one relay process on the
  host, a sidecar next to my app, or a relay container on the
  BlueField — and what is on the DPU side that actually terminates
  the connection?"*. Answered by the three-axis configuration model
  (deployment shape × socket type × forwarding endpoint) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ## configure`](TASKS.md#configure) step 2.
- **"My relay is up but the host application cannot connect."** —
  worked example: *"the relay process is running but the host app
  reports `ECONNREFUSED` / connect timeout when it tries the
  socket / port the relay should be listening on"*. Answered by the
  layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  layers 1-3 + the bind / accept ladder in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"Bytes leave the host but never arrive on the DPU side."** —
  worked example: *"the host app's socket connect succeeded, the
  relay reports the connection accepted, but the DPU-side
  terminator never sees the data"*. Answered by the
  forwarding-endpoint layer in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  layer 4 + the silent-data-path failure mode named in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"How do I prove the relay is the right answer before I admit
  the whole fleet onto it?"** — worked example: *"I have N host
  clients; I want to confirm one of them works end-to-end before
  pointing the rest at the relay"*. Answered by the
  smoke-before-bulk loop in
  [`TASKS.md ## test`](TASKS.md#test) (bind → confirm one host app
  connects → confirm one round-trip end-to-end → only then admit
  the fleet).
- **"Is this Socket Relay shipped on my installed DOCA, and does
  its version match the comch / eth pieces it sits on?"** — worked
  example: *"is the relay binary present on this host, and does
  it agree with `pkg-config --modversion doca-common`"*. Answered
  by the version overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) rules and adds the
  Socket Relay specifics (presence check, host vs BlueField
  packaging, agreement with companion libraries).

## Audience

This skill serves **external operators, application owners, and AI
agents who need to bridge a socket-oriented application onto a
BlueField DPU without rewriting the application against the DOCA
programming surface**. Concretely:

- A platform owner with an existing host service that speaks a
  socket protocol to a peer and wants the peer to live on the
  BlueField instead — without porting the host service onto
  [`doca-comch`](../../libs/doca-comch/SKILL.md).
- A migration engineer evaluating the relay as the *first* step of
  a phased move onto DOCA, with the comch / RDMA / Ethernet
  rewrite reserved for a later phase.
- An SRE / platform operator deploying the relay as a service
  container on the BlueField via the runtime contract documented
  in [`doca-container-deployment`](../../doca-container-deployment/SKILL.md).
- An AI agent answering *"my host app cannot reach the DPU on the
  socket I configured — what do I check"* with the relay's layered
  error surface as the diagnosis ladder.

It is **not** for users debugging the Socket Relay binary itself,
**not** a substitute for the live public DOCA Socket Relay guide,
and **not** the right place for users learning the comch
programming API or for users doing line-rate raw-packet I/O. Those
audiences belong in [`doca-comch`](../../libs/doca-comch/SKILL.md)
and [`doca-eth`](../../libs/doca-eth/SKILL.md) respectively.

The Socket Relay is shipped as a documented DOCA artifact on
installs that include it; depending on the operator's deployment
shape, it can be invoked as a CLI on the host or BlueField Arm, or
deployed as a service container on the BlueField via the
documented kubelet-standalone runtime in
[`doca-container-deployment`](../../doca-container-deployment/SKILL.md).
The skill uses the same `kind: tool` three-file shape as the
rest of the bundle so the agent's task-verb contract
(`configure / build / modify / run / test / debug`) is uniform
across libraries, services, and tools.

## When to load this skill

Load this skill when the user is — or the agent needs to — drive
the DOCA Socket Relay on a real host or BlueField Arm with DOCA
installed (or inside the public NGC DOCA container with the right
device passthrough). Concretely:

- Migrating a socket-based application off a same-host peer onto
  a BlueField-side terminator without rewriting the application.
- Picking the relay's deployment shape (in-process beside the app,
  sidecar process / container, or a service container on the
  BlueField) for a specific environment.
- Configuring the host-side socket the relay binds and the DPU-side
  forwarding endpoint the relay points at, before any application
  client tries to connect.
- Walking the bind → connect → round-trip → admit-fleet loop on a
  brand-new relay deployment.
- Diagnosing a *"host app cannot connect"*, *"connection accepted
  but no bytes flow"*, or *"first round-trip works, the rest hang"*
  symptom against the relay's layered error surface.
- Cross-checking the relay's view against the host application
  side and the DPU-side terminator when the three appear to
  disagree.

Do **not** load this skill for general DOCA orientation, the comch
programming API, RDMA programming, line-rate raw packet I/O, or
DOCA install. For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-comch`](../../libs/doca-comch/SKILL.md),
[`doca-eth`](../../libs/doca-eth/SKILL.md), or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what the Socket Relay carries and changes:
  the three-axis configuration model (deployment shape × socket
  type / protocol × forwarding endpoint), the read-only vs
  state-changing operation split, the version-availability overlay
  that redirects to [`doca-version`](../../doca-version/SKILL.md),
  the layered error taxonomy (tool-not-installed / relay-not-bound
  / host-app-not-connecting / DPU-side-terminator-not-reachable /
  permission / version / cross-cutting), the relay's role as the
  data-plane counterpart to
  [`doca-comch`](../../libs/doca-comch/SKILL.md), and the
  high-stakes safety policy that makes a misconfigured forwarding
  endpoint a silent data-path break.
- `TASKS.md` — step-by-step workflows for the in-scope task verbs:
  `configure` (the three-axis decision + the precondition probe),
  `build` (route to install — the relay is shipped pre-built),
  `modify` (refuse — modify the *invocation / deployment*, not the
  binary), `run` (the bind → connect → round-trip flow), `test`
  (the smoke-before-bulk eval loop), `debug` (the layered
  diagnosis ladder), plus a `Deferred task verbs` block and a
  `Command appendix` that honors the bundle's
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  preamble.

The skill assumes a host or BlueField where DOCA is already
installed (or the public NGC DOCA container is running with the
right device passthrough) and the operator has whatever privileges
the public DOCA Socket Relay guide requires for the chosen
deployment shape.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle.
To keep the boundary clean, it deliberately does not contain — and
pull requests should not add:

- **Verbatim binary names, flag inventories, subcommand names,
  socket-path defaults, port numbers, or output column names.**
  The public DOCA Socket Relay guide on `docs.nvidia.com` and the
  installed `--help` on the user's version are the joint source
  of truth; copying them here pins the skill to one release and
  silently rots when the relay evolves. The skill routes the agent
  at those sources instead.
- **Pre-baked example output.** Output is install-, version-, and
  deployment-specific. A captured example will mislead an operator
  on a different platform / state.
- **Wrappers, parsers, or scripts** in any language that consume
  the relay's output. The output format is documented; users who
  want to script against it should read the live guide and write
  the parser against their installed version.
- **A `samples/` or `reference/` subtree.** This is a thin loader
  for a documented DOCA artifact; substantive material lives on
  the public page and in `--help`.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (the user wants to bridge a socket-based application onto
   the DPU without rewriting it onto the DOCA programming surface,
   not learn the comch / eth / RDMA APIs).
2. **For what the relay carries, the three-axis configuration
   model, the read-only vs state-changing split, version
   availability, the layered error surface, observability, and
   safety posture, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the smoke-before-bulk
   workflow — `configure`, `build`, `modify`, `run`, `test`,
   `debug`, plus the `Command appendix` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-comch`](../../libs/doca-comch/SKILL.md) — the host ↔ DPU
  **control-plane** primitive (PCIe-based message channel between a
  host and a DPU process). The Socket Relay is the **data-plane**
  counterpart for socket-oriented applications: comch is what an
  application uses to *coordinate* across the host ↔ DPU boundary
  programmatically; the relay is what an application uses to
  *carry* socket-shaped bytes across that boundary without being
  rewritten. Pair the two when an application owner is migrating
  in stages — control plane first via comch, data plane via the
  relay until a per-library rewrite is justified.
- [`doca-eth`](../../libs/doca-eth/SKILL.md) — the line-rate raw
  packet I/O surface that sits *below* the socket level. The relay
  presents a socket-shaped interface to the application and rides
  the underlying DOCA fabric; doca-eth is the right answer when
  the application can be rewritten to operate on packets directly
  rather than sockets, and when the throughput / latency profile
  the relay achieves is no longer sufficient.
- [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  — the BlueField service-runtime contract (kubelet standalone +
  static-pod manifests + per-service config-file mount). Load this
  alongside the present skill when the operator's chosen
  deployment shape is a relay container running on the BlueField
  rather than a host-side process; the runtime contract there owns
  the pod-spec / image-pull / static-pod / liveness pieces, this
  skill owns the relay-specific config.
- [`doca-comm-channel-admin`](../doca-comm-channel-admin/SKILL.md)
  — the operator-side admin CLI for comch channels. The
  list → inspect → decide pattern there is the same shape the
  Socket Relay uses for its own state: read-only inspection first,
  state-changing operations gated on a clean smoke. Both tools are
  members of the same family and the same patterns apply.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Socket Relay guide and the rest of
  the public DOCA documentation set. The canonical URL is reached
  via [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. The `## Version compatibility` section
  in [`CAPABILITIES.md`](CAPABILITIES.md) is a concise overlay
  that redirects here for the body.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back → report contract for
  structured helper tools. The Command appendix in
  [`TASKS.md`](TASKS.md) honors this contract.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, representor visibility checks, and the *I
  have no install yet* path with the public NGC DOCA container.
  This skill assumes its preconditions are satisfied.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. The Socket Relay slots in at the *runtime* layer
  as a data-path bridge whose layered error surface escalates to
  the cross-cutting ladder when the cause is below DOCA.
