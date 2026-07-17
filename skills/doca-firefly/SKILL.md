---
license: Apache-2.0
name: doca-firefly
description: >
  Use this skill when the user is operating the DOCA Firefly Service
  container on BlueField — picking the four PTP configuration axes
  (role / profile / domain / interface), wiring the BlueField PHC +
  host follower + consumer workload pairing, deciding whether
  PTP-grade time is even needed (vs. chrony / NTP), or debugging a
  Firefly deployment where PTP isn't syncing or the host clock isn't
  following. Trigger even when the user does not explicitly mention
  "DOCA Firefly" or "PTP" — typical implicit phrasings include
  "container green but PTP never advances past LISTENING", "Firefly
  says synced but the host clock still drifts", "sync acquired but
  offset is tens of microseconds", "my Rivermax SMPTE workload needs
  PTP", or "is chrony good enough". Refuse and route elsewhere for
  installing DOCA, host-side chrony / ptp4l config bodies, PTP
  topology / boundary-clock design, building DOCA apps that read the
  disciplined PHC, or other DOCA services (DMS, Flow-Inspector, HBN)
  — those belong to other skills.
metadata:
  kind: service
compatibility: >
  BlueField-Arm-only DOCA service container; pulled from NVIDIA NGC
  and started under the BlueField OS container runtime. Host-side
  install is irrelevant. Requires a reachable PTP master (or runs as
  the master itself) and a PTP-aware network path; the host-side
  time follower (chrony / ptp4l / phc2sys reading the BlueField PHC)
  is also operator-owned.
---

# DOCA Firefly Service

> **Subsystem inventory (Run-12 correction, verified Run-13).**
> DOCA Firefly is NOT just "a PTP daemon." The shipped
> `doca_firefly.yaml` exposes **six** PTP-stack subsystems via
> environment variables, each with its own `*_STATE`,
> `*_CONFIG_FILE`, and (where relevant) `*_INTERFACE` /
> `*_DEVICE` knobs (the count is six because the PTP Monitor
> subsystem ships an internal `phc2sys` monitor client that is
> distinct from the standalone PHC2SYS subsystem — both ship in
> the same container image):
>
> 1. **PTP** (`PTP_STATE`, `PTP_INTERFACE`, `PTP_CONFIG_FILE`) —
>    the `ptp4l` daemon (or master, depending on profile) that
>    drives the BlueField PHC.
> 2. **PTP Monitor** (`MONITOR_STATE`, `MONITOR_CONFIG_FILE`,
>    `MONITOR_CLIENT_TYPE`, `MONITOR_CLIENT_PHC2SYS_INTERFACE`,
>    `MONITOR_CLIENT_CONNECTION_TIMEOUT`) — the monitor server +
>    client surface; the **internal `phc2sys` monitor client**
>    (`MONITOR_CLIENT_TYPE=phc2sys`) is a real subsystem inside
>    Firefly, not just a host-side concern.
> 3. **PHC2SYS** (`PHC2SYS_STATE`, `PHC2SYS_ARGS`,
>    `PHC2SYS_CONFIG_FILE`) — the **container-internal** `phc2sys`
>    instance; the bundle previously framed `phc2sys` as
>    host-only, which is wrong.
> 4. **PPS** (`PPS_STATE`, `PPS_DEVICE`) — the Pulse-Per-Second
>    output (with the additional `enable_while_running` and
>    `do_nothing` states beyond plain enable/disable).
> 5. **SyncE** (`SYNCE_STATE`, `SYNCE_INTERFACE`,
>    `SYNCE_CONFIG_FILE`) — Synchronous Ethernet frequency
>    distribution; orthogonal to PTP.
> 6. **Firefly Servo** (`SERVO_STATE`, `SERVO_CONFIG_FILE`) —
>    the proprietary Firefly servo loop (alternative to the
>    upstream linuxptp servo).
>
> The valid `PROFILE` values are exactly **`default` / `media` /
> `telco-l2` / `custom`** (per `doca_firefly.yaml` comments) —
> the agent must not invent additional values. All five
> subsystems respond to `defined_by_profile` so the active
> `PROFILE` is what actually flips most of the knobs.
>
> Configuration-override env vars follow the pattern
> `CONF_<SUBSYSTEM>_<section>_<key>` (e.g.
> `CONF_PTP_global_priority1`, `CONF_SYNCE_global_backend`,
> `CONF_MONITOR_global_telemetry_export`); these are the
> documented surface for overriding individual config keys
> without shipping a full custom config file.

**Where to start:** This skill is for *operating* the DOCA Firefly
Service container, not for *linking against* a library. Firefly is
the **PTP / PHC2SYS / PPS / SyncE / Servo / Monitor** stack that
drives and observes the BlueField PTP Hardware Clock (PHC); it is
*not* the host-side time follower, *not* the consumer workload, and
*not* a programming surface. If the user wants to *deploy* the
container, open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure). If the question is *what shape
of service is Firefly and what PTP roles / profiles does it speak*,
start at [`CAPABILITIES.md`](CAPABILITIES.md). If DOCA is not installed
on the BlueField yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user's
real question is *"I have a Rivermax SMPTE workload and the docs say I
need PTP"*, the right pairing is this skill **plus**
[`doca-rmax`](../../libs/doca-rmax/SKILL.md) — Firefly disciplines
the PHC; Rivermax reads the disciplined time.

## Example questions this skill answers well

The CLASSES of Firefly questions this skill is built to answer, each
with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **"Do I actually need Firefly, or is NTP / chrony good enough?"** —
  worked example: *"my distributed app is fine on chrony today; is
  there a reason to add PTP?"*. Answered by the PTP-vs-NTP path-
  selection rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"What four PTP configuration axes do I have to decide before
  starting the container?"** — worked example: *"a SMPTE ST 2110
  broadcast plant that wants Firefly in slave role on the wire-side
  port"*. Answered by the four-axis configuration table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the PTP-config step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Firefly's container is running but the host's time isn't
  following — what did I miss?"** — worked example: *"`ptp4l` /
  Firefly says it's locked but `chronyc tracking` on the host shows
  drift"*. Answered by the END-TO-END time-sync discipline in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the host-follower step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"PTP locks but the offset / jitter is way past spec — what's
  wrong with the path?"** — worked example: *"sync acquired but offset
  is in the tens of microseconds"*. Answered by the PTP-aware-path
  rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the layered debug ladder in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"How does Firefly pair with a Rivermax SMPTE workload?"** —
  worked example: *"SMPTE ST 2110 video sender that needs to be PTP-
  locked"*. Answered by the Rivermax-pairing rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the pairing step in
  [`TASKS.md ## configure`](TASKS.md#configure), which routes the
  Rivermax side to
  [`doca-rmax`](../../libs/doca-rmax/SKILL.md) and refuses to
  collapse the two services into one.
- **"My Firefly container starts but PTP never reaches
  `SLAVE` / `MASTER` state — was it role, domain, profile, or
  interface?"** — worked example: *"container green but the
  ports-state output never advances past `LISTENING`"*. Answered by
  the four-axis-mismatch rule in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug).

## Audience

This skill serves **external operators and platform teams who deploy
the DOCA Firefly Service container** to provide PTP-grade time
synchronization to time-sensitive workloads on BlueField + the host
behind it. Concretely: people running the Firefly container on
BlueField Arm, choosing its PTP role / profile / domain / interface
from the public Firefly guide, wiring the host-side follower (chrony
with the PHC source, or `ptp4l` reading the PHC) so the host clock
tracks the BlueField PHC, and validating the end-to-end discipline
  before scaling a Rivermax, 5G UPF, financial-trading, or distributed-
database workload that depends on it.

It is **not** for NVIDIA developers contributing to Firefly itself,
and it is **not** a programming guide for *building applications on
top of* DOCA libraries (that is
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
plus the matching `libs/<library>` skill). Firefly is a **service**,
not a library: the operator runs a container and configures PTP via
the documented config surface; they do not link against a
`libfirefly.so` to write their own program.

**Path selection up front.** Use Firefly when sub-microsecond,
PTP-grade time precision is required on BlueField AND the host (SMPTE
ST 2110 broadcast workloads layered on Rivermax, 5G UPF time
requirements, distributed systems that need PTP-grade time, anything
where NTP / chrony jitter is not tight enough). Do **not** reach for
Firefly when NTP / chrony already meets the workload's time-precision
budget, when no PTP-aware switching / boundary-clock infrastructure
exists in the path, or when pure software-side time precision is
sufficient — in those cases the correct answer is to keep the host's
existing chrony / NTP setup and route the agent away from Firefly,
not to deploy it speculatively.

## When to load this skill

Load this skill when the user is doing **hands-on Firefly deployment
work** on a BlueField where DOCA is already installed. Concretely:

- Deciding *whether* Firefly is the right answer for the user's
  time-precision requirement (vs. keeping NTP / chrony on the host).
- Deploying the Firefly container on BlueField Arm — choosing image
  source per the public DOCA Firefly Service Guide, mounting the
  Firefly config, and starting / stopping the container.
- Choosing the four PTP configuration axes — PTP role (master /
  slave / boundary clock / transparent clock), profile (the
  `PROFILE` env var accepts EXACTLY `default` / `media` /
  `telco-l2` / `custom` per `services/firefly/doca_firefly.yaml`;
  these map onto industry PTP profile names: `default` → IEEE 1588,
  `media` → SMPTE 2059-2, `telco-l2` → G.8275.1 only (G.8275.2
  corresponds to the separate `telco-l3` config, reached via
  `custom`) — do NOT put the industry names directly into the env
  var), domain number,
  network interface — for the user's deployment.
- Wiring the host-side follower so the host clock tracks the
  BlueField PHC (chrony with the PHC source, or `ptp4l` /
  `phc2sys` reading the PHC) — without this step the host clock
  does NOT follow the Firefly-disciplined PHC, regardless of how
  cleanly Firefly comes up.
- Pairing Firefly with a time-sensitive consumer workload (Rivermax
  SMPTE, 5G UPF, finance, distributed databases) and validating
  the end-to-end discipline.
- Reading the Firefly container's logs, the PHC offset, the
  ports-state output, or any other documented observability surface
  to confirm PTP is locked.
- Debugging a Firefly deployment where the container is healthy but
  PTP is not syncing, or PTP is syncing but the host clock is not
  following, or sync is up but jitter is past spec.

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, library-API questions, or non-PTP time topics. For
those, route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-setup`](../../doca-setup/SKILL.md), or the matching
`libs/<library>` skill.

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — Firefly's architecture (container that drives
  the BlueField PHC and speaks PTP on the wire), the four PTP
  configuration axes (role / profile / domain / interface, with
  transport as a fifth knob), the deployment shape (container on
  BlueField Arm per the public Container Deployment Guide), the
  pairing surface (Rivermax + host-side time-sync follower), the
  observability surface (container logs + PHC offset + ports state),
  the error taxonomy (four-axis-mismatch / host-follower / PTP-aware-
  path / container-runtime), and the safety policy (PTP-vs-NTP path
  selection, END-TO-END discipline, smoke-before-scale).
- `TASKS.md` — step-by-step workflows for the in-scope Firefly
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`,
  plus a `Deferred task verbs` block routing out-of-scope questions
  and a `Command appendix` of recurring commands.

The skill assumes a BlueField where DOCA is already installed and
the operator has the privileges the public Firefly Service Guide
expects to pull, run, and configure containers on BlueField Arm.
It does not cover installing DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or sample-config
bundle. To keep the boundary clean, it deliberately does not contain —
and pull requests should not add:

- **Pre-baked Firefly configuration files** (full PTP config blocks,
  ready-to-run role / profile / domain bundles) intended to be
  copy-pasted into production. PTP configuration is deployment-
  specific (per the user's profile, domain plan, interface naming,
  and upstream PTP topology); the safe answer for an external
  operator is to derive the config from the public Firefly Service
  Guide against their own deployment. The agent's job is to
  prescribe the *procedure* and the *four-axis decision*, not to
  ship a config the user might run unmodified.
- **Container image names, tags, or registry paths.** The
  authoritative image source is the public DOCA Firefly Service
  Guide reachable through
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services);
  Firefly's image tag is version-bound and changes between DOCA
  releases. Inventing or memorizing a tag is the canonical
  hallucination failure mode for a service skill.
- **Host-side chrony stanzas or `ptp4l` / `phc2sys` config files.**
  Those are host-environment-specific and live on the host, not
  inside the Firefly container. The skill names *that* the
  host-side follower must be wired and *what its source must be*
  (the BlueField PHC); the chrony / `ptp4l` config bodies belong
  to the host operator and to upstream Linux PTP documentation.
- **A `samples/`, `templates/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled *"reference"*, is misleading: operators will read it
  as production-ready.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope **and** that Firefly is the right answer at all (vs. keeping
   NTP / chrony on the host).
2. **For Firefly's deployment shape, the four PTP configuration
   axes, the Rivermax + host-follower pairing surface, the error
   taxonomy, the observability surface, and the END-TO-END safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public DOCA Firefly Service Guide and
  the rest of the public DOCA documentation set. The Firefly URL is
  listed under
  [`## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation and
  install verification on the BlueField where the Firefly container
  will run, including the *I have no install yet* path via the
  public NGC DOCA container. This skill assumes its preconditions
  are satisfied on BlueField Arm.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. Firefly's container tag is version-bound;
  this skill's `## Version compatibility` cross-links the four-way
  match rule and adds the container-tag-lags-host-package overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer /
  fall back / report). The Command appendix in [TASKS.md](TASKS.md)
  honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA patterns. Firefly is service-shaped not library-
  shaped, so the build / modify / first-app pattern there does not
  apply directly, but the cross-library debug discipline (frontend-
  before-backend, env-before-program) remains useful when Firefly
  reports an error that originated in the container runtime or in
  a DOCA library it called.
- [`doca-rmax`](../../libs/doca-rmax/SKILL.md) — the canonical
  paired workload. SMPTE ST 2110 Rivermax streams depend on a
  Firefly-disciplined PHC; Firefly is the time-source side and
  Rivermax is the timing-precise data-plane side. The two skills
  load together for any broadcast-style deployment, and they do
  NOT collapse into one another — Firefly does not stream media;
  Rivermax does not discipline the PHC.
- [`doca-dms`](../doca-dms/SKILL.md) — sibling service skill. The
  agent reading both skills should see the same service-skill shape
  (container, BlueField Arm, deployment pattern, smoke-before-scale,
  env preconditions, config schema) layered on top of a different
  per-service domain (DMS = device management via gNMI / gNOI;
  Firefly = time synchronization via PTP).
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Firefly-specific debug (PTP not syncing,
  host clock not following, jitter past spec) overlays on top of
  that ladder.
