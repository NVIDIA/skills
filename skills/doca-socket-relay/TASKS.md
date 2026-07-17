# DOCA Socket Relay — Tasks

**Where to start:** The verbs that carry real workflow content are
`## configure`, `## run`, `## test`, and `## debug`. The other two
substantive verbs (`build`, `modify`) carry routing stubs because the
Socket Relay is a shipped binary / container image, not a source
artifact the external operator compiles or patches — the *thing the
agent modifies* is the relay's invocation and deployment, not the
relay itself. The `## test` verb is an iterative loop (bind → connect →
round-trip → admit fleet, with every state-changing operation re-opening
the smoke), not a one-shot pass — see the eval-loop overlay in
`## test` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through the six
task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), then explicitly
defers task verbs that do not belong here, and ends with the
`Command appendix` honoring the bundle's
[`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
preamble.

For the Socket Relay, the verbs that carry real workflow content are
`configure`, `run`, `test`, and `debug`. The other two verbs *exist as
anchors* because the agent's task-verb contract is uniform across
libraries, services, and tools — and each one carries a meaningful
**routing stub** that names where the user's question really belongs.

## configure

The Socket Relay's *configuration* is the three-axis decision documented
in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
deployment shape × socket type / protocol × forwarding endpoint —
layered on top of the precondition that DOCA is installed on the side
the operator is about to invoke the relay from. The relay does not
carry a separate config file or daemon contract of its own beyond what
the chosen deployment shape pulls in (a service-container deployment
inherits the per-service config-file mount contract from
[`doca-container-deployment CAPABILITIES.md ## Capabilities and modes`](../../doca-container-deployment/CAPABILITIES.md#capabilities-and-modes)).

Steps the agent should walk the user through, in order:

1. **Axis 1 — pick the deployment shape.** Commit explicitly to one of
   the three shapes named in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   *in-process* beside the host application, *sidecar* process /
   container next to the application on the same host, or *container*
   service container on the BlueField via the kubelet-standalone runtime.
   An answer that picks one shape without naming the alternatives has
   silently narrowed the relay to a single deployment story; surface
   the trade-offs (host-OS permission surface for in-process; lifecycle
   coupling for sidecar; the static-pod / image-pull / per-service
   config-file mount surface from
   [`doca-container-deployment ## configure`](../../doca-container-deployment/TASKS.md#configure)
   for the container shape).
2. **Axis 2 — confirm the host-leg socket type.** The shipped
   `doca_socket_relay` binary uses AF_UNIX (Unix Domain Sockets,
   SOCK_STREAM) on its host leg as of DOCA 3.3 — the `-s/--socket`
   argument is a filesystem path, not an IP/port. If the user's
   wrapping application speaks TCP or UDP, the relay does not
   transparently bridge that — the user either changes the wrapping
   app to talk AF_UNIX or uses a different transport. Confirm what
   the relay binary actually supports on the installed version by
   running `doca_socket_relay --help` on the host before committing
   to a transport plan; do not invent transport modes the binary
   does not register.
3. **Axis 3 — pick the forwarding endpoint on the DPU.** Name the
   DPU-side terminator the relay forwards to (the relay's far half, a
   DPU-side service, or a documented BlueField service container that
   re-presents the bytes to the DPU peer). This is the most
   consequential axis for safety — a wrong forwarding endpoint
   produces a *silent* data-path break per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   and is its own layer in
   [`## debug`](#debug). Do not commit to an endpoint that the agent
   has not confirmed exists from the DPU side.
4. **Confirm the DOCA host install and version compatibility.** Walk
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
   to capture the installed DOCA version on the side the relay will
   run, then apply the Socket Relay overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility):
   confirm the relay artifact is shipped on this install profile and
   that its version agrees with the underlying `doca-common`. For the
   container-shape deployment, add the third anchor (the per-service
   container tag) per
   [`doca-container-deployment CAPABILITIES.md ## Version compatibility`](../../doca-container-deployment/CAPABILITIES.md#version-compatibility).
   If the artifact is not present on the install profile, route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
5. **Identify the wrapping host application's pre-relay behavior.**
   The smoke in [`## test`](#test) compares the application's
   *with-relay* behavior to its *no-relay* baseline; the agent should
   capture the baseline before any state-changing operation so the
   regression check is grounded. If the application has no pre-relay
   baseline (e.g. the DPU-side peer never existed without the relay),
   surface that — the regression check degrades into a one-sided
   liveness check and the agent should say so.
6. **Plan the rollback path.** Because every state-changing relay
   operation sits in the data path per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   the agent records: the pre-relay configuration the application used
   (which socket / port / path), the previous-known-good relay
   invocation (or a no-relay baseline) ready to re-apply, and a way to
   reach the operator if the deploy disrupts the host application. For
   the container-shape deployment, the rollback path layered on top is
   the static-pod-file removal documented in
   [`doca-container-deployment ## modify`](../../doca-container-deployment/TASKS.md#modify).

For the canonical DOCA universal lifecycle that underlies any
DOCA-library work the wrapping host application might also do once
the migration moves past the relay phase (per the staged-migration
framing in
[`SKILL.md ## Related skills`](SKILL.md#related-skills)),
see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill is concerned with the *operator-side* configuration of the
relay deployment, not the program-side lifecycle of any DOCA library a
later phase introduces.

## build

The Socket Relay binary / container image is **shipped pre-built** as
part of every DOCA install / NGC catalog entry that includes it; there
is no source tree the external operator is expected to compile, no
build flags, no `meson` or `make` workflow for the relay itself.

What the operator *does* build, in some cases, is the **wrapping host
application** — the socket-oriented service the operator is migrating
onto the DOCA fabric via the relay. The build context for that
application is the application's own existing build (the relay is
intentionally transparent to it); the relay's purpose is to let the
wrapping application stay buildable as-is. The agent should not invent
a relay-specific build wrapper for an application whose existing build
already produces a working binary against a local socket — the whole
point of the relay is to leave that binary untouched.

Routing for nearby "build" questions:

- *"The relay binary / image isn't there — do I need to build it?"* →
  no. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  for the install path that ships the relay (and
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container alternative). For the
  container-shape deployment, route the image-pull / NGC tag lookup
  through
  [`doca-container-deployment ## configure`](../../doca-container-deployment/TASKS.md#configure)
  step 4.
- *"I want to build my wrapping host application against DOCA so I can
  drop the relay later."* → not a relay-build question. Route to
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the cross-library build pattern and the matching `libs/<library>`
  skill ([`doca-comch ## build`](../../libs/doca-comch/TASKS.md#build)
  for the control-plane rewrite,
  [`doca-eth`](../../libs/doca-eth/SKILL.md) for the line-rate
  packet-I/O rewrite) for the library-specific overlay. The relay
  carries socket-shaped bytes today; the per-library build is what
  retires it later.
- *"I want to build a custom DPU-side terminator that talks to the
  relay's far half."* → not a relay-build question. The DPU-side peer
  is either a documented BlueField service container (route to
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md))
  or a program written against the matching DOCA library
  ([`doca-comch`](../../libs/doca-comch/SKILL.md) for control-plane
  bridging, [`doca-eth`](../../libs/doca-eth/SKILL.md) for line-rate
  data); route accordingly.
- *"I want to extend the relay with a new transport / scenario."* →
  out of scope here; this skill is for external operators consuming the
  shipped relay, not for contributors extending it.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) explicitly forbids adding a build recipe or
wrappers for the Socket Relay; revisit that policy before changing this
section.

## modify

**Do not modify the shipped Socket Relay binary / container image.** It
is an NVIDIA-shipped artifact; there is no documented public way to
change its behavior, output format, or operation surface, and none
should be invented.

What the agent *does* modify, every time, is the **relay deployment** —
the deployment shape, the socket / forwarding-endpoint configuration,
the invocation flags the public guide documents on the installed
version. Two recurring modify shapes apply, and **each is a HIGH-STAKES
change** per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
that the agent must gate on a fresh
[`## test`](#test) smoke:

1. **Change the deployment shape without changing the forwarding
   contract.** Move the relay from in-process to sidecar, from sidecar
   to a BlueField service container, or back, while keeping the
   forwarding endpoint and socket type / protocol the same from the
   host application's point of view. The application's view of *what it
   is talking to* is unchanged, but the precondition surface flips
   (host-OS permissions for in-process; lifecycle coupling for sidecar;
   the full static-pod / image-pull / per-service config-file mount
   contract from
   [`doca-container-deployment ## modify`](../../doca-container-deployment/TASKS.md#modify)
   for the container shape). Re-walk
   [`## configure`](#configure) axis 1 and the precondition step, then
   re-open the
   [`## test`](#test) smoke loop before declaring the change done.
2. **Change the forwarding endpoint without changing the deployment
   shape.** Repoint the relay at a different DPU-side terminator (a
   different DPU-side service, a different BlueField service container,
   a different address on the same terminator). The deployment-shape
   precondition surface is unchanged but the *silent data-path break*
   risk is at its highest — the host application keeps connecting to
   the same relay endpoint while its bytes are quietly delivered
   somewhere new (or nowhere). Re-walk
   [`## configure`](#configure) axis 3 with the new endpoint confirmed
   reachable *from the DPU side first*, then re-open the
   [`## test`](#test) round-trip smoke before any other host client is
   admitted.

Routing for nearby "modify" questions:

- *"The output format / log shape is inconvenient — can I change it?"*
  → no, not inside this skill. The documented surface is the surface;
  if the user wants structured output, the right answer is *"check
  whether the installed version exposes one per `--help`, otherwise
  write a parser against the documented format on your installed
  version"* — and even the parser is out of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"I need a different *kind* of bridging than the relay provides
  (programmatic control over channels, line-rate raw packet I/O)."* →
  re-examine axis 1 of the design space: the relay is one half of the
  host ↔ DPU bridging picture, and the right answer may be a per-
  library rewrite via
  [`doca-comch`](../../libs/doca-comch/SKILL.md) (control plane) or
  [`doca-eth`](../../libs/doca-eth/SKILL.md) (line-rate data plane)
  rather than a relay re-configuration.
- *"Can I patch the relay to add a feature?"* → out of scope; this
  skill is for consumers of the shipped relay, not contributors to it.

## run

The smoke-before-bulk flow — every relay deployment goes through it,
no exceptions. The full invocation surface lives in the public DOCA
Socket Relay guide reachable via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
this section names the *shape* of the flow, not the verbatim command
lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent binary names, flag strings, socket-path defaults, or
port numbers"*).

1. **Confirm the binary / image, version, and deployment shape are in
   place.** Per [`## configure`](#configure) steps 1-4; without this,
   the next four steps will burn the operator's time on a deployment
   that the install does not actually support. For the container-shape
   deployment, add the
   [`doca-container-deployment ## run`](../../doca-container-deployment/TASKS.md#run)
   steps 1-3 (image pulled, pod-spec dropped, kubelet shows `Running`)
   as preconditions to step 2 below.
2. **Bind the relay on its configured endpoint.** Start the relay on
   the side the chosen deployment shape names (in-process or sidecar
   on the host; service container on the BlueField); read the relay's
   own *bound / listening* signal from its `--help`-documented
   inspection surface or the container ENTRYPOINT log per
   [`doca-container-deployment CAPABILITIES.md ## Observability`](../../doca-container-deployment/CAPABILITIES.md#observability).
   A bind failure here is layer 2 in
   [`## debug`](#debug); do not move on until the relay reports bound
   against the configured socket / port / path.
3. **Confirm one host application client can connect.** Bring up *one*
   host application client (not the fleet) and verify, from the
   application side and the relay side, that the connect succeeded:
   the application's own connect / send / recv error reporting reads
   clean, the relay's documented connection-set inspection lists the
   client. A *"the relay accepts but no client appears"* asymmetry
   here is layer 3 in
   [`## debug`](#debug); a *"the application reports
   `ECONNREFUSED` / connect timeout"* is layer 2 or 3 depending on
   which side the misconfiguration sits on.
4. **Confirm one end-to-end round-trip.** Drive a single, trivial
   round-trip from the host application through the relay to the
   DPU-side terminator and back — one request the operator already
   knows the expected response shape for. Capture *all three* views:
   the relay's *byte received from the host* signal, the DPU-side
   terminator's *byte delivered* signal (per its own
   observability surface, e.g.
   [`doca-comm-channel-admin TASKS.md ## run`](../doca-comm-channel-admin/TASKS.md#run)
   when the DPU-side peer is comch-backed, or
   [`doca-container-deployment ## run`](../../doca-container-deployment/TASKS.md#run)
   step 4-5 when the peer is a BlueField service container), and the
   host application's *response received* signal. A *"connection
   accepted but no bytes arrive"* asymmetry here is the canonical
   silent-data-path break in
   [`## debug`](#debug) layer 4.
5. **Only after steps 2-4 read clean** may the agent admit the rest of
   the host application fleet onto the relay. The fleet admit is *one
   client at a time* on a HIGH-STAKES relay (per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy));
   batching the fleet onto a freshly-bound relay makes failure
   attribution much harder.
6. **Stop in reverse order.** Drain or stop the host application
   clients first; only then tear down the relay; only then tear down
   the DPU-side terminator. Tearing the relay or terminator down while
   the host application is connected produces socket disconnects /
   `ECONNRESET` / partial-byte cross-version corruption that masks
   unrelated bugs in subsequent runs.

When recording the run for downstream consumers (the *baseline*
pattern), write down: the DOCA version (per
[`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)),
the deployment shape, the side(s) the relay runs on, the socket type /
protocol the relay binds, the forwarding endpoint the relay points at,
and the full unredacted relay-side / host-side / DPU-side captures from
step 4. The downstream `## test` and `## debug` workflows depend on
those six fields.

## test

The Socket Relay's `## test` verb is about *testing the bridge*, not
unit-testing the relay binary. The bench-style measurement story belongs
to
[`doca-bench`](../doca-bench/SKILL.md); this skill's test loop is the
*operational* smoke that proves the relay-mediated path actually carries
the host application's traffic.

**`## test` is an iterative loop, not a one-shot pass.** Every state-
changing operation on the relay — a re-bind, a forwarding-endpoint
repoint, a deployment-shape change, a container restart, a wrapping-app
restart that triggers reconnects — re-opens the smoke. Treating it as
a one-shot pass is the failure mode this loop replaces.

The smoke-before-bulk shape (three substantive checks; each is
load-bearing):

1. **Smoke = one round-trip.** Drive a single, trivial request through
   the host application → relay → DPU-side terminator and confirm the
   reply path back. Quote *all three* observability surfaces named in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) —
   relay-side, host-application-side, DPU-side terminator — not just
   the application's *"it worked"* impression. A one-sided pass is half
   the evidence; the cross-check is what closes it.
2. **Topology check.** Trace the path explicitly: host application →
   relay's host-side bind → relay's far half / DPU-side container →
   DPU-side terminator → reply path back to the host application. Each
   arrow has its own observability surface per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability);
   the topology check is satisfied when every arrow shows a matching
   *byte in / byte out* event for the same round-trip. A blank
   observability surface on any arrow is itself a finding — the agent
   surfaces *which arrow* is dark, rather than guessing.
3. **Regression baseline — the no-relay direct-host comparison.** Per
   the safety rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   before admitting the host application fleet to a relay-mediated
   path in production, run the same trivial request against the same
   DPU-side peer **without** the relay — via the application's
   pre-relay socket path if it still exists, or via a direct connection
   to the DPU-side terminator if the deployment supports it. A relay
   that *only* works with the relay in the picture, but does not
   reproduce the application's pre-relay behavior, is hiding a
   regression that the round-trip smoke alone cannot expose. If a
   no-relay baseline is not feasible in the operator's environment,
   the agent must say so — the regression check degrades to a
   one-sided liveness check and the agent flags the gap, rather than
   silently quoting the round-trip as proof of regression-freedom.

Eval-loop overlay (rows apply to every relay deployment, not just
one):

| Iteration trigger | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| Smoke round-trip succeeded; second client hangs | Could be a fleet-admit limit, a socket-resource limit, or a per-connection asymmetry | Re-walk [`## run`](#run) step 5 admit cadence; if still wedged, route to [`## debug`](#debug) layer 3 (host-app-not-connecting) before adding more clients. |
| Smoke round-trip never closes; the relay says "accepted" but the application sees nothing | Forwarding-endpoint misconfiguration — the silent data-path break | Jump to [`## debug`](#debug) layer 4 (DPU-side-terminator-not-reachable); do NOT retry the round-trip until the forwarding endpoint is re-verified from the DPU side. |
| Regression baseline disagrees with the with-relay path | The relay introduced a behavior change, OR the baseline path itself drifted | Stop iterating on relay tuning; capture both observability snapshots, route to [`## debug`](#debug) layer 4 + 6 (version) before any state-changing operation. |
| Round-trip succeeds on host A, fails on host B at the same DOCA version | Deployment-shape precondition or host-OS permission delta | Re-walk [`## configure`](#configure) step 1 + the precondition surface; route the permission piece to [`## debug`](#debug) layer 5. |
| Same invocation works on DOCA version X, breaks on Y | Version layer; *is* a regression signal — provided the deployment-shape and forwarding-endpoint configs are captured on both sides | Cross-link the two baselines, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 and [`doca-container-deployment ## debug`](../../doca-container-deployment/TASKS.md#debug) layer 7 for the container shape. |
| Relay reports zero / hung after a quiet period | Could be the relay (state machine), the DPU-side terminator (peer gone), or below (driver, BlueField mode) | Stop iterating on the wrapping-app side; jump to [`## debug`](#debug) layers 4 + 7. |

The agent's rule: every state-changing action on the relay re-opens the
smoke. Re-running a tweaked configuration and quoting *"the round-trip
came back"* without re-checking the topology arrows and the no-relay
baseline is exactly the failure mode this loop replaces — on a
HIGH-STAKES relay the cost of that failure mode is silent data-path
breakage of the entire wrapping-application fleet, not just *"weird
traffic"*.

This skill does **not** ship a "test fixture" or pre-recorded expected
output. The expected output is install-, version-, deployment-shape-,
and wrapping-application-specific; pinning one would mislead operators
on a different platform / state. See
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## debug

When the Socket Relay fails to bind, the host application cannot
connect, the round-trip never closes, or the regression baseline
disagrees with the with-relay path, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order. The shape of the diagnosis:

1. **Tool-not-installed (layer 1).** The relay artifact is not present
   on the side the operator is invoking it from — no relay binary, no
   relay container image, no relay subpackage. Confirm DOCA is
   installed on that side via
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure),
   confirm the install profile includes the relay per the
   public guide reached through
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools),
   and apply the Socket Relay overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
   Route to [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   if absent.
2. **Relay-not-bound — port / socket layer (layer 2).** The relay
   process / container starts but cannot bind the host-side socket /
   port / UDS path the operator configured. The relay's own message is
   ground truth; the agent quotes it verbatim rather than guessing.
   Re-quote the configuration from the public guide and `--help` on
   the installed binary; route filesystem / privilege issues to the
   host-OS team and to
   [`doca-setup CAPABILITIES.md ## Safety policy`](../../doca-setup/CAPABILITIES.md#safety-policy);
   for the container-shape deployment, walk
   [`doca-container-deployment ## debug`](../../doca-container-deployment/TASKS.md#debug)
   layer 5 (volume-mount) when the bind target is a host-path UDS
   mounted into the container.
3. **Host-app-not-connecting (layer 3).** The relay is bound on its
   configured endpoint but the host application reports it cannot
   reach the relay. Reconcile the application-side configuration with
   the relay-side bound endpoint *before* declaring either side
   broken — quote both. Common causes: the application is configured
   for a different socket / port / path than the relay actually bound,
   the application and the relay sit in different network / mount
   namespaces (especially with the container-shape deployment), or a
   host-firewall rule is dropping the connect. The right answer when
   the application is configured against an endpoint the relay never
   bound is to fix one side or the other consistently, not to bind a
   second relay on the *application*'s endpoint and silently fan out.
4. **DPU-side-terminator-not-reachable (layer 4) — the silent
   data-path break.** The host application connects to the relay
   successfully, the relay accepts, **and yet bytes never arrive at
   the DPU-side peer**. This is the canonical high-stakes failure mode
   per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   The agent's diagnosis discipline:
   - Capture the relay's view (was the byte received from the host?),
     the BlueField-side terminator's view (was the byte delivered?),
     and the DPU-side peer application's view (was the byte
     processed?) before blaming any single layer.
   - If the DPU-side peer is a documented BlueField service container,
     walk
     [`doca-container-deployment ## debug`](../../doca-container-deployment/TASKS.md#debug)
     to confirm the pod is `Running` and the ENTRYPOINT log is clean.
   - If the DPU-side peer is a comch-backed program, walk
     [`doca-comm-channel-admin TASKS.md ## debug`](../doca-comm-channel-admin/TASKS.md#debug)
     to confirm the channel side is healthy.
   - A retry of *"connect again"* without root-cause analysis is the
     wrong move; that is exactly the silent-break failure mode this
     layer exists to surface.
5. **Permission layer (layer 5).** The relay runs and accepts the
   configured options but reports it cannot bind the socket, open the
   UDS path, or join the network namespace it needs because the
   invoking user lacks the privileges the public guide names. The
   relay's own message is ground truth; the fix is to re-run with the
   correct privileges per the public guide and the host-OS rules, not
   to bypass the check. Route container-shape permission issues to
   [`doca-container-deployment ## debug`](../../doca-container-deployment/TASKS.md#debug)
   layer 5.
6. **Version layer (layer 6).** The relay runs but its behavior
   disagrees with what the public guide on the screen describes — a
   documented flag is rejected, a documented connection-state field
   never appears, or the relay's `--version` and `pkg-config
   --modversion doca-common` disagree (partial install). Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 end-to-end and apply the Socket Relay overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
   for the container-shape deployment, add the third anchor per
   [`doca-container-deployment CAPABILITIES.md ## Version compatibility`](../../doca-container-deployment/CAPABILITIES.md#version-compatibility).
7. **Cross-cutting layer (layer 7).** The relay is bound, the host
   application connects, the forwarding endpoint is verified reachable,
   the version four-way match passes — and the symptom remains. The
   cause is below the relay: kernel / driver, BlueField mode, firmware,
   host networking misconfiguration, or hardware. Escalate to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured relay-side inspection plus the host-side and
   DPU-side traces as evidence; do not loop on bind / unbind / restart
   at this layer hoping for a different outcome.

In every case: **quote what the relay said, and quote both ends of the
bridge.** Do not paraphrase the relay's connection-set output, do not
"summarize" the round-trip into prose, do not blame one side of the
bridge without quoting the other side's view. The relay sits in the
data path precisely so that an inspection of both sides is feasible;
collapsing it to one side wastes the inspection.

## Deferred task verbs

The following verbs are out of scope for this skill but are commonly
asked in the same conversations. Route them as follows so the agent
does not invent guidance:

- **Large-scale relay orchestration** (managing many relay endpoints
  across a fleet of host applications and BlueField DPUs, multi-tenant
  fan-out, relay-aware connection routing) ⇒ out of scope here; deferred
  to the operator's own platform tooling. This skill teaches the
  per-deployment three-axis decision and the smoke-before-bulk loop,
  not a multi-cluster control plane. The bundle does not currently
  ship a fleet-orchestration skill; route platform-team conversations
  to the operator's existing platform.
- **Writing a custom DPU-side terminator from scratch** ⇒ not a
  relay-tool question. The DPU-side peer the relay forwards to is
  *its own program*, written against the matching DOCA library —
  [`doca-comch`](../../libs/doca-comch/SKILL.md) when the operator
  wants programmatic control over the host ↔ DPU bridge, or
  [`doca-eth`](../../libs/doca-eth/SKILL.md) when the operator needs
  line-rate raw packet I/O. The relay carries the bytes; the
  per-library skill is where the terminator's program-side lifecycle
  lives.
- **Container packaging of the relay (image build, tag policy,
  static-pod authoring)** ⇒ out of scope here. The container-shape
  deployment of the relay rides on the shared BlueField service
  container runtime documented in
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md);
  the pod-spec shape, image-pull procedure, static-pod manifests
  directory, and per-service config-file mount are owned by that skill.
  This skill names *that* the container shape is one of the three
  documented deployment shapes; the runtime contract belongs there.
- **Installing DOCA on the host or BlueField target** ⇒ out of scope
  here. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure) (and
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path). The relay is shipped by
  the install; this skill does not own the install workflow.
- **General host ↔ DPU control-plane programming** ⇒ not a relay-tool
  question. Route to
  [`doca-comch`](../../libs/doca-comch/SKILL.md) and to
  [`doca-comm-channel-admin`](../doca-comm-channel-admin/SKILL.md) for
  the operator-side admin pattern on the comch side.

## Command appendix

Socket Relay-specific invocation classes the verbs above reach for.
Every row is a CLASS — the agent must not invent binary names, flag
strings, subcommand names, socket-path defaults, port numbers, or
output column names beyond what the public DOCA Socket Relay guide and
`--help` on the installed artifact document. The
read-only-first / state-changing-after-smoke symmetry is the
load-bearing piece.

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
| Detect that the relay artifact is present on this side | The documented relay-presence probe on the chosen deployment shape — the documented binary location plus the `--help` flag on the installed CLI, or the documented image-list / image-inspect command for the container shape per [`doca-container-deployment ## Command appendix`](../../doca-container-deployment/TASKS.md#command-appendix) | [`## configure`](#configure) step 4; [`## debug`](#debug) layer 1 | The artifact reports its documented `--help` output (or the image-inspect returns the documented per-service tag); absence routes to [`doca-setup ## install`](../../doca-setup/TASKS.md#configure). |
| Confirm relay version against the DOCA fabric | The documented binary's own `--version` flag (per `--help`), cross-checked with `pkg-config --modversion doca-common` per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure) | [`## configure`](#configure) step 4; [`## test`](#test) cross-check + [`## debug`](#debug) layer 6 | Both strings agree under the four-way match in [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility); disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). |
| Bind the relay on its configured host-side endpoint | The documented relay-bind invocation on the chosen deployment shape — invocation shape comes from the public guide and the installed `--help`; for the container shape, the documented pod-spec drop per [`doca-container-deployment ## modify`](../../doca-container-deployment/TASKS.md#modify) step 6 | [`## run`](#run) step 2 | The relay reports its documented *bound / listening* signal against the configured socket / port / path; a bind failure routes to [`## debug`](#debug) layer 2. |
| Observe the relay's current channel / connection set | The shipped `doca_socket_relay` binary does NOT expose a dedicated `--list-channels` / `--list-connections` flag in its argp registration as of DOCA 3.3 — visibility is via the DOCA logger output (logger category `SOCKET_RELAY`) at `--sdk-log-level INFO` or `DEBUG`. Capture log lines around connection acceptance, message forwarding, and disconnects to derive the current connection set. The agent quotes column names from `--help` on the installed binary first, and does not invent inspection flags that the binary does not register. | [`## run`](#run) step 3 + [`## test`](#test) step 1; [`## debug`](#debug) layer 3 | Exit 0; the logger lines include the host application client the operator expected, with the connection-state transitions for a healthy active client. |
| Drive the round-trip probe end-to-end | A trivial host-application-side request whose expected response shape the operator already knows, cross-checked against the relay's connection-set inspection (row above) and the DPU-side terminator's own observability surface (per [`doca-comm-channel-admin TASKS.md ## run`](../doca-comm-channel-admin/TASKS.md#run) for a comch-backed terminator, or [`doca-container-deployment ## run`](../../doca-container-deployment/TASKS.md#run) for a service-container terminator) | [`## run`](#run) step 4 + [`## test`](#test) step 1 | All three observability surfaces (host-app-side, relay-side, DPU-side terminator) report the matching *byte in / byte out* events for the same round-trip; a blank surface routes to [`## debug`](#debug) layer 4. |
| Diff the with-relay path against a no-relay baseline | The same trivial round-trip from row above, re-run against the host application's pre-relay socket path (or a direct host connection to the DPU-side terminator if the deployment supports it); the relay is removed for the comparison run per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) | [`## test`](#test) step 3 | The host application's observed behavior matches between the with-relay and no-relay runs (response shape, latency order of magnitude, error rate); a divergence routes to [`## debug`](#debug) layer 4 + 6 before any state-changing operation. |
| Route to the cross-cutting debug ladder | Capture the relay-side inspection + the host-application-side trace + the DPU-side terminator's view as one artifact bundle, then hand off per [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug); cross-cutting env commands (`pkg-config --modversion`, `dmesg`, `mlxconfig -d <bdf> q`, representor enumeration) live in [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix) and [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix) | [`## debug`](#debug) layer 7 | The downstream debug ladder consumes the three-view artifact bundle as evidence; looping on bind / unbind / restart at this layer is explicitly the wrong move per the safety rule. |

Three cross-cutting rules for this appendix:

- **Never invent a binary name, flag string, subcommand name,
  socket-path default, port number, or output column name.** `--help`
  on the installed artifact plus the public DOCA Socket Relay guide
  via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
  are the joint contract; prose-derived strings are the most common
  hallucination failure for this skill.
- **State-changing operations re-open the smoke.** A re-bind, a
  forwarding-endpoint repoint, a deployment-shape change, a container
  restart — each is HIGH-STAKES per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  and re-opens the
  [`## test`](#test) round-trip + topology + regression-baseline loop.
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`pkg-config --modversion`, `dmesg`, `mlxconfig -d <bdf> q`,
  representor enumeration, port state) live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix)
  and
  [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  the container-shape deployment runtime commands live in
  [`doca-container-deployment TASKS.md ## Command appendix`](../../doca-container-deployment/TASKS.md#command-appendix);
  this appendix names only Socket Relay-specific invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file, restated here
so they are visible at the point of action and not buried in
[`SKILL.md`](SKILL.md):

- The **public DOCA Socket Relay guide** plus the installed `--help`
  are the joint source of truth. When they disagree (e.g. a flag
  landed in a release this skill was not written against), the
  *installed* `--help` wins for the operator's actual run.
- **Read-only operations are safe**; **state-changing operations are
  not**. The agent must say which class an operation belongs to before
  recommending it, and must gate every state-changing operation on a
  clean round-trip + topology + regression-baseline smoke per
  [`## test`](#test).
- **A misconfigured forwarding endpoint silently breaks app
  connectivity.** This is the canonical high-stakes failure mode per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy);
  the agent surfaces it explicitly before any forwarding-endpoint
  change.
- **Quote what the relay said, both ends of the bridge.** The
  three-view artifact bundle (relay-side, host-application-side,
  DPU-side terminator) is the unit a relay finding is meaningful in;
  paraphrasing or summarizing loses fidelity the rest of the bundle's
  procedures depend on.
- This skill **assumes a healthy DOCA install** on the side the relay
  runs (or the public NGC DOCA container) and whatever privileges the
  public guide names for the chosen deployment shape. If the install
  is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) before running anything
  else here. For the host ↔ DPU control-plane sibling whose channels
  coordinate the endpoints the relay carries traffic between, see
  [`doca-comch`](../../libs/doca-comch/SKILL.md) and its operator-side
  admin counterpart
  [`doca-comm-channel-admin`](../doca-comm-channel-admin/SKILL.md).
