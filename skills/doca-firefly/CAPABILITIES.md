# DOCA Firefly Service — Capabilities

**Where to start:** The pattern overview below names the recurring
Firefly-class operational patterns. Pick the pattern first, then
drill into the H2 that owns the substance. For the *how* of executing
each pattern, jump to [TASKS.md](TASKS.md).

This file enumerates Firefly's documented capabilities, deployment
shape, configuration axes, and operational behaviors as described in
the public DOCA Firefly Service Guide. Treat it as a *map of what is
documented*, not a substitute for reading the live page when
configuring a real deployment. For the public URL itself, route
through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
— this skill does not duplicate the URL routing.

## Pattern overview

Every Firefly-class question this skill teaches resolves into one of
FIVE patterns. The patterns are CLASSES — they apply across every
Firefly deployment, not just one PTP role or one consumer workload.

| Firefly pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide whether Firefly is the right answer | PTP-grade time precision vs. chrony / NTP good enough; PTP-aware path available or not | [`## Safety policy`](#safety-policy) path-selection rule |
| 2. Pick the four PTP configuration axes | PTP role + profile + domain + interface (transport as fifth knob) — every axis is a mismatch hazard | [`## Capabilities and modes`](#capabilities-and-modes) four-axis table |
| 3. Wire the END-TO-END time-sync discipline | Firefly disciplines the BlueField PHC; the host clock follower (chrony / `ptp4l`) reads the PHC; the consumer workload reads the disciplined time | [`## Safety policy`](#safety-policy) END-TO-END rule + [`## Capabilities and modes`](#capabilities-and-modes) deployment shape |
| 4. Pair with a time-sensitive consumer | Rivermax SMPTE, 5G UPF, finance, distributed databases — each pairs with Firefly in the same shape (Firefly = time source; consumer = time reader) | [`## Capabilities and modes`](#capabilities-and-modes) pairing table |
| 5. Map a Firefly symptom back to its layer | Container-runtime vs. four-axis mismatch vs. host-follower vs. PTP-aware-path-vs-jitter — four independent layers, each with its own owner | [`## Error taxonomy`](#error-taxonomy) layered split |

Two cross-cutting rules that apply to *every* pattern above:

- **PTP is end-to-end; Firefly only owns the BlueField PHC.** The
  most common first-time-deploy failure is *"Firefly says it's
  locked, but my host clock / workload still drifts"*. That is by
  construction: Firefly disciplines the PHC; the host clock and the
  consumer workload each need their own follower wired to the PHC.
  Pretending Firefly alone makes the whole stack PTP-correct is the
  load-bearing misconception this skill exists to prevent.
- **Operate the documented path; do not invent one.** Firefly's
  config schema, container image source, PTP role names, profile
  names, and observability surface are all documented in the public
  DOCA Firefly Service Guide. Quoting config keys, image tags, or
  CLI flags not in the public guide is the most common hallucination
  failure mode for this skill.

## Capabilities and modes

### Service shape

Firefly is a **long-running container** that ships from NGC and runs
on the BlueField Arm cores. The container is the daemon: it owns the
PTP state machine, drives the BlueField PHC, and speaks PTP on the
configured network interface to upstream and downstream PTP peers.
There is no host-side Firefly binary the user installs — Firefly is
the container; the host's relationship to Firefly is to read the
disciplined PHC via the host's own time-sync follower (chrony or
`ptp4l` / `phc2sys`).

Three architectural properties the operator must hold throughout:

- **The PHC is the boundary.** Firefly drives the BlueField PTP
  Hardware Clock; everything else (BlueField OS clock, host OS
  clock, consumer workload clock) is downstream of the PHC and
  must be disciplined to follow it. Firefly does NOT discipline the
  host clock directly.
- **The container is the unit of deployment.** Operators do not
  start `firefly` as a host binary; they start the Firefly
  container per the public Container Deployment Guide pattern
  (same shape as every other DOCA service container — see the
  sibling [`doca-dms`](../doca-dms/SKILL.md) for the same shape on
  a different per-service domain).
- **PTP behavior is configured by file, not by CLI flag.** The
  container's PTP role, profile, domain, interface, and transport
  are set in the documented Firefly config file the operator
  mounts into the container. The agent should NOT invent CLI flag
  names; the config file is the contract.

## Deployment shape

The public Firefly Service Guide documents the container deployment
on BlueField Arm. The shape lines up with every other DOCA service
container — pull from NGC, mount the config, start under the
documented runtime (the BlueField OS's container manager per the
public Container Deployment Guide). For the canonical container-
deployment recipe shared with the other DOCA service containers,
route through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

Two deployment-shape rules:

- **BlueField Arm only.** Firefly is a BlueField-side service; it
  does not run on the host. The host's relationship to Firefly is
  via the network (PTP on the wire) and via the PHC (time follower).
- **One Firefly per BlueField, on the right port.** Firefly drives
  the BlueField PHC for the BlueField as a whole; the user picks
  *which* network interface carries PTP traffic (typically the
  wire-side port, not the management port). Running two Firefly
  containers fighting over the same PHC is a configuration error,
  not a redundancy strategy.

### Four-axis PTP configuration

Every Firefly deployment must commit to four PTP configuration axes
before starting the container. Get any one wrong and PTP does not
sync — the symptom is *"container green, PTP never advances past
`LISTENING`"*, and the diagnosis walks the four axes in order. The
axes are jointly documented in the public Firefly Service Guide;
quote the exact valid values from there rather than from memory.

| Axis | Class shape | Mismatch symptom | Where to look |
| --- | --- | --- | --- |
| **PTP role** | Master (best clock on segment); slave (follows an upstream master); boundary clock (slave-on-one-port + master-on-another, propagates time across segments); transparent clock (forwards PTP with residence-time correction) | Container starts, ports-state never advances; or two masters on the same segment fight | Public Firefly Service Guide's PTP role section |
| **Profile** | `PROFILE` env var accepts EXACTLY `default` / `media` / `telco-l2` / `custom` (per `services/firefly/doca_firefly.yaml` comments, verified in source). These four env-var values map onto the industry-standard PTP profile names: `default` → IEEE 1588 default profile; `media` → SMPTE 2059-2 (broadcast); `telco-l2` → ITU-T G.8275.1 ONLY (telecom full timing over L2 / Ethernet; the bundled `ptp4l-telco-l2.conf` extends `G.8275.1.cfg`); `custom` → whatever the user's custom config file specifies. ITU-T G.8275.2 (telecom partial timing) corresponds to the separate `telco-l3` config (`ptp4l-telco-l3.conf` extends `G.8275.2.cfg`), which is NOT an accepted `PROFILE` env-var value and is reached only via `custom`. Do NOT put the industry-standard names (`SMPTE 2059-2`, `G.8275.1`, `G.8275.2`, `IEEE 1588`) directly into the `PROFILE` env var — those are NOT accepted env-var values | Profiles silently disagree with the upstream master; sync acquires then breaks | `services/firefly/doca_firefly.yaml` PROFILE comments + Public Firefly Service Guide's profile section |
| **Domain number** | PTP allows multiple time domains on the same network; only peers on the same domain see each other | Container green, *zero* PTP peers seen even though `tcpdump` shows PTP frames on the wire | Public Firefly Service Guide's domain section |
| **Network interface** | Which BlueField port carries PTP — typically the wire-side port, not the management port; must be a port with hardware PTP timestamping enabled on the NIC | Container green, no PTP traffic egresses; or PTP traffic egresses but jitter is huge because timestamping is software-only | Public Firefly Service Guide's interface section |
| Transport (fifth knob) | L2 multicast / L3 unicast / L3 multicast; must match the upstream PTP topology | Frames on the wire but never matched by upstream — the upstream is L2 multicast and Firefly is configured L3 unicast (or vice versa) | Public Firefly Service Guide's transport section |

The agent's rule: **the four-axis decision precedes everything
else**. A deployment that starts the container before the operator
can name the role, profile, domain, and interface is going to debug
the wrong axis first. Force the decision up front.

### Pairing with time-sensitive consumers

Firefly is the time-source side of every supported pairing. The
consumer workload is the time-reader side. Both sides must be wired;
Firefly alone is not a finished deployment.

| Consumer workload | Why it needs Firefly | Pairing shape |
| --- | --- | --- |
| Broadcast SMPTE ST 2110 on Rivermax | SMPTE 2110 mandates PTP-locked time on every endpoint that emits or receives video / audio / ancillary streams; the Rivermax SDK assumes the PHC is being disciplined externally | Firefly drives the PHC with the SMPTE 2059-2 profile; the Rivermax-using workload reads the disciplined PHC via [`doca-rmax`](../../libs/doca-rmax/SKILL.md); the host clock follower wires chrony or `ptp4l` to the PHC so the host OS clock also follows. See [`doca-rmax CAPABILITIES.md ## Safety policy`](../../libs/doca-rmax/CAPABILITIES.md#safety-policy) for the Rivermax-side precondition matrix |
| 5G UPF (5G User Plane Function) | 5G timing requirements (G.8275.1 / G.8275.2 telecom profiles) demand PTP-grade time | Firefly drives the PHC with the G.8275.x profile; the UPF workload reads disciplined time from the host clock follower wired to the PHC |
| Financial trading | Regulatory time-stamping requirements (e.g. sub-microsecond time accuracy on trade records) | Firefly drives the PHC with the user's chosen profile; trading process reads disciplined time |
| Distributed databases needing high-precision time | Conflict ordering, distributed transactions, and externally consistent reads benefit from PTP-grade time | Same shape — Firefly drives PHC; DB process reads disciplined time |

The agent's rule: when the user mentions Rivermax / SMPTE / 5G UPF /
finance / distributed time precision, name Firefly *and* the
consumer pairing *and* the host-side follower in the same breath.
Naming only one of the three is how end-to-end discipline gets
silently broken.

### Configuration model

The Firefly container is configured by a documented config file
that the operator mounts into the container at the path the public
guide names. The config file declares the four-axis PTP
configuration (role, profile, domain, interface) plus the transport
choice and any advanced PTP knobs the user's profile expects. Quote
config keys from the live public Firefly Service Guide; do not infer
them from generic `ptp4l` or `linuxptp` knowledge — Firefly's config
schema is documented per the public guide and may not be 1:1 with
upstream PTP daemons.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Firefly-specific overlay** is:

- **Firefly is a NGC container; the container tag is the runtime
  version anchor.** Same pattern as
  [`doca-dms`](../doca-dms/SKILL.md): the Firefly container ships
  from NGC with its own tag that may lag the host's DOCA package
  version, and the relevant version anchor for an as-deployed
  Firefly is the container tag pulled, not `pkg-config --modversion`
  on the host. Always quote both versions when the user reports a
  Firefly behavior; if they diverge, route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before diagnosing the Firefly behavior itself.
- **Profile support is version-bound.** Profiles like SMPTE 2059-2 /
  G.8275.1 / G.8275.2 are jointly conditional on the Firefly
  container version and the BlueField OS / firmware version. When
  the user asks *"does this profile work on my deployment?"*, the
  authoritative answer is the public Firefly Service Guide page
  whose version matches the container tag pulled.
- **Read the public Firefly Service Guide version header.** The
  guide is versioned; the on-page version must match the container
  tag the operator is using. A mismatch between the docs version
  and the container tag is the canonical *"my config doesn't work
  even though it matches the docs"* failure mode.

## Error taxonomy

Firefly errors fall into five layers, each with its own owner. The
agent's rule: walk the layers in order; do NOT skip down without
clearing the layer above.

| Layer | Symptom | Root cause class | Where to fix |
| --- | --- | --- | --- |
| 1. Container runtime | Container fails to start, restart-loops, exits immediately, image pull fails | Image tag wrong, registry credentials missing, BlueField runtime not configured to run this container, config file mount path wrong | BlueField container runtime + the public Container Deployment Guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| 2. Four-axis PTP config | Container green, PTP never advances past `LISTENING`; or no peers seen on the wire; or wrong upstream master selected | One or more of the four axes (role / profile / domain / interface) mismatches the upstream PTP infrastructure | [`## Capabilities and modes`](#capabilities-and-modes) four-axis table; the fix is config, not container |
| 3. Host-side follower | PTP locks (Firefly says synced; PHC offset is tight); host OS clock drifts; `chronyc tracking` or `date` shows seconds-of-drift | Host-side chrony / `ptp4l` / `phc2sys` not configured to follow the PHC; the BlueField PHC is in sync but the host has not been told to read it | Host's chrony / `ptp4l` / `phc2sys` configuration (upstream Linux PTP / chrony docs) — Firefly is correct |
| 4. PTP-aware path | PTP locks; sync acquired; offset and jitter are wildly past the profile's spec | A non-PTP-aware switch is in the path between Firefly and its peer; the switch silently adds variable latency that PTP cannot correct for | Network-side fix: insert boundary clocks at the non-PTP-aware switch, or replace the switch with a PTP-aware one — Firefly cannot fix a non-PTP-aware path from the endpoint |
| 5. Consumer workload | Firefly correct; PHC correct; host clock follows PHC; but the consumer workload still reports time drift | The workload reads its own time source instead of the disciplined system clock; e.g., a Rivermax workload that was never wired to the disciplined PHC | The consumer-side skill — for Rivermax, [`doca-rmax CAPABILITIES.md ## Safety policy`](../../libs/doca-rmax/CAPABILITIES.md#safety-policy); for 5G UPF or a distributed DB, the workload's own docs |

The agent's rule: **never recommend a Firefly config change without
first identifying which of the five layers is the cause**. The most
common debug failure for this skill is misreading a layer-3 symptom
(host clock drifts) as a layer-2 problem (Firefly config) and
rewriting the Firefly config when the fix is on the host.

## Observability

Documented observability surfaces the agent should reach for, in
order of how cheaply they answer the *"is Firefly actually working"*
question:

1. **Container state.** First — is the Firefly container actually
   running? The BlueField container manager reports container
   status, restart count, and the container's stdout / stderr log
   stream. A restart loop is a layer-1 (container runtime) symptom
   per [`## Error taxonomy`](#error-taxonomy); diagnose it before
   touching PTP config.
2. **Firefly's own logs.** The container's stdout (and any
   documented log destination the public guide specifies) is the
   primary Firefly observability surface. Look for the PTP state
   machine's transitions (`LISTENING → UNCALIBRATED → SLAVE`,
   `LISTENING → MASTER`, etc.) and for the documented error /
   warning lines. The agent should NOT invent log line formats;
   quote what the live container is emitting.
3. **PHC offset and frequency.** The BlueField PHC's offset
   (relative to its master) and frequency adjustment are the
   numeric proof that PTP is disciplining the clock. The PHC is a
   Linux-kernel PTP clock and is read with the standard upstream
   PTP tooling (e.g. `pmc -u -b 0 'GET CURRENT_DATA_SET'`,
   `phc_ctl /dev/ptpN get`); the agent should defer to upstream
   Linux PTP documentation for the exact invocation rather than
   memorize one.
4. **Host-side follower status.** Whatever follower the host is
   running (chrony with the PHC source, or `ptp4l` / `phc2sys`)
   exposes its own status — `chronyc tracking`, `chronyc sources`,
   `pmc` against the host-side PTP daemon. A healthy
   Firefly + unhealthy host follower = layer-3 symptom per the
   error taxonomy.
5. **PTP-on-the-wire confirmation.** When the agent suspects a
   four-axis mismatch (layer 2), the cheapest confirmation is a
   `tcpdump` for PTP traffic on the configured interface — does
   Firefly egress PTP frames at all, are any PTP frames ingressing
   from a master, do the domain numbers in the frames match the
   configured domain. The agent should defer to upstream Linux PTP
   / `tcpdump` documentation for the exact filter; the load-bearing
   point is *"go look at the wire before changing config a fifth
   time"*.

For the cross-library debug-time observability (`DOCA_LOG_LEVEL`,
`--sdk-log-level`, the trace build flavor — relevant when Firefly
calls into a DOCA library that emits structured logs), see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Firefly's safety surface is **path-selection first**, then the
END-TO-END discipline, then the smoke-before-scale rule, then the
operational disciplines around the container itself.

- **Path-selection rule (load-bearing).** Firefly is the right
  answer only when PTP-grade time precision is *actually* required
  AND a PTP-aware path is *actually* available. Concretely:
    - Use Firefly when the consumer workload (SMPTE ST 2110 on
      Rivermax, 5G UPF, financial trading, sub-microsecond
      distributed-database time) genuinely needs PTP-grade time
      precision AND the network between Firefly and the PTP master
      is PTP-aware (PTP-aware switches in the path, or boundary
      clocks where the switches are not PTP-aware).
    - Do NOT reach for Firefly when chrony / NTP already meets the
      workload's time-precision budget (which is typical for normal
      workloads), when no PTP-aware switching / boundary-clock
      infrastructure exists in the path, or when pure software-side
      time precision is sufficient. In those cases the right answer
      is to keep the host's existing chrony / NTP setup and
      explicitly tell the user *"Firefly is the wrong tool here;
      here's why"* — not to deploy it speculatively and end up
      debugging PTP that was never needed.
- **END-TO-END discipline (load-bearing).** Firefly disciplines the
  BlueField PHC. The host clock follower (chrony with the PHC
  source, or `ptp4l` / `phc2sys` reading the PHC) is the operator's
  responsibility. The consumer workload's wiring to the disciplined
  time (e.g. Rivermax reading the PHC) is also the operator's
  responsibility. A Firefly deployment that names only the
  BlueField side and stops there is a deployment that will *fail
  silently at the host or workload*, not a deployment that works.
  The agent must always teach the three legs together: BlueField
  PHC + host clock follower + consumer workload.
- **Smoke before scale.** Before pointing the time-sensitive
  consumer workload at the disciplined PHC, the agent must walk the
  user through a smoke: Firefly container running, PTP advanced to
  `SLAVE` / `MASTER`, PHC offset within profile spec, host
  follower's `chronyc tracking` / `pmc` status reporting *Reference
  ID = PHC* and a tight offset. Only then layer the consumer
  workload on top. A workload that comes up before the smoke passes
  silently uses a wrong time source, and the bisection across
  Firefly / host / workload is much harder.
- **One Firefly per BlueField on one interface.** Two Firefly
  containers on the same BlueField fighting over the same PHC is
  a configuration error; the agent must NOT recommend it as a
  redundancy strategy. PTP redundancy is a network-side concern
  (multiple PTP masters, BMCA election) that does not require
  multiple Firefly containers.
- **Don't paper over a non-PTP-aware path.** When the symptom is
  *"sync acquired but jitter is way past spec"* and the layer is
  *"a non-PTP-aware switch in the path"* per
  [`## Error taxonomy`](#error-taxonomy), the honest answer is
  *"the network path doesn't support the precision you asked for;
  the fix is a PTP-aware switch or boundary clocks, not a Firefly
  config knob"*. Silently turning down the user's precision
  expectation, or pretending a Firefly knob can fix a non-PTP-aware
  switch, is a user-visible regression dressed up as helpfulness.

## Public-source pointer

The single canonical public source for Firefly is the **DOCA
Firefly Service Guide**, reachable through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
Verify that the version of the guide matches the Firefly container
tag pulled on the BlueField — Firefly's config surface, supported
profiles, and observability output are documented to evolve, so
config keys and profile names can change between releases.
