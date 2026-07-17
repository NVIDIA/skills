# DOCA Firefly Service — Tasks

**Where to start:** The order is `configure → build → modify → run →
test → debug`. The `## test` verb is an iterative loop, not a
one-shot pass — see the eval-loop overlay in `## test` below. For
Firefly, `build` and `modify` are about *deployment configuration*
(container image selection, mounted config file, host-follower
wiring), not about compiling source.

These verbs cover the in-scope Firefly operational workflows for an
external operator deploying the Firefly container on BlueField.
Every step assumes the operator has consulted the live public DOCA
Firefly Service Guide (reachable through
[doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services))
and is using it as the authoritative reference; this file prescribes
the *order* and *what to look up where*, not a copy-paste runbook.

## configure

Preparing the BlueField, picking the four PTP axes, and wiring the
end-to-end discipline before the container starts.

1. **Confirm Firefly is actually the right answer.** Per the
   path-selection rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
    - Is PTP-grade time precision genuinely required by the consumer
      workload (SMPTE ST 2110 on Rivermax, 5G UPF, financial
      trading, or sub-microsecond distributed-database time)?
    - Is the network between Firefly and the upstream PTP master
      PTP-aware (PTP-aware switches in the path, or boundary clocks
      where the switches are not PTP-aware)?
    - If either answer is *no*, stop here and tell the user
      honestly: keep the host's chrony / NTP setup, *do not* deploy
      Firefly speculatively.
2. **Confirm the env is healthy.** This skill expects DOCA to be
   installed on the BlueField. If that has not been verified, run
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test) first.
   If the user has no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path.
3. **Decide the four PTP configuration axes.** Per the four-axis
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   commit before starting the container to:
    - **Role** — master / slave / boundary clock / transparent
      clock. Drives whether Firefly follows or sources time on the
      segment.
    - **Profile** — the `PROFILE` env var accepts EXACTLY `default` /
      `media` / `telco-l2` / `custom` (per
      `services/firefly/doca_firefly.yaml`). These four values map
      onto the industry-standard PTP profile names: `default` →
      IEEE 1588 default; `media` → SMPTE 2059-2 (broadcast);
      `telco-l2` → G.8275.1 only (telecom full timing over L2;
      `ptp4l-telco-l2.conf` extends `G.8275.1.cfg`); `custom` → the
      user's own config file (e.g. G.8275.2 partial timing
      corresponds to the separate `telco-l3` config, which is NOT an
      accepted `PROFILE` value and is reached only via `custom`). Do
      NOT put the industry-standard names directly into the env var
      — they are not accepted values. Must match (or interop with)
      the upstream master's profile.
    - **Domain number** — must match the upstream master's domain;
      mismatched domains = zero peers visible.
    - **Network interface** — typically the wire-side BlueField
      port, not the management port. The port must have hardware
      PTP timestamping enabled.
    - Plus the transport choice (L2 multicast / L3 unicast / L3
      multicast) per the upstream topology.
4. **Plan the host-side follower.** Decide which mechanism the host
   will use to read the BlueField PHC — chrony with the PHC source,
   or `ptp4l` / `phc2sys` reading the PHC. Without this step the
   host clock does NOT follow the Firefly-disciplined PHC, no
   matter how cleanly the container comes up. Capture the device
   path the host will read (`/dev/ptpN`) and the follower's config
   intent; the body of the host-side config file is the host
   operator's responsibility and lives in upstream Linux PTP /
   chrony docs, not in Firefly.
5. **Plan the consumer workload pairing.** If the deployment exists
   to support a Rivermax SMPTE workload, the Rivermax-side
   precondition matrix (Rivermax SDK + license + scheduling
   discipline) is ALSO required — route the Rivermax side to
   [`doca-rmax ## configure`](../../libs/doca-rmax/TASKS.md#configure).
   For 5G UPF / finance / distributed databases, the consumer-side
   wiring is owned by the workload's own docs; Firefly's job is to
   keep the disciplined time available.
6. **Author the Firefly container config.** From the public DOCA
   Firefly Service Guide, derive the config file fragment for the
   chosen role / profile / domain / interface / transport. Quote
   config keys from the live guide, do NOT infer them from generic
   `ptp4l` knowledge. Plan where the config file will live on the
   BlueField filesystem and what mount path the container expects.

## build

Firefly is a service shipped as a container, not a library. There is
no Firefly *application* artifact for the operator to build — the
container ships from NGC and the config is a static file.

If the user is asking how to build a **PTP client** in their own
language (e.g. an application that reads the disciplined system
clock with sub-microsecond precision), that is not a Firefly
question:

- For applications that **read the BlueField PHC directly** via a
  DOCA library (e.g. a Rivermax-based SMPTE workload), the build is
  the DOCA library's build — route to
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  and the matching `libs/<library>` skill (e.g.
  [`doca-rmax`](../../libs/doca-rmax/SKILL.md) for SMPTE).
- For applications that **read the OS system clock** that the
  host-side follower is disciplining, no DOCA-specific build is
  needed — the application uses standard `clock_gettime(2)` against
  the appropriate clock, and the precision is whatever the
  host-side follower delivers. The follower setup is upstream
  Linux PTP / chrony, not Firefly.

If the user is instead asking how to build the **Firefly container
itself** from source, that is *not* an external-operator workflow —
the container ships pre-built from NGC and rebuilding it is out of
scope for this skill. Route to the public DOCA Firefly Service
Guide via [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).

## modify

Firefly does not have a "modify a sample" workflow analogous to DOCA
libraries; there is no Firefly sample program a user starts from.
The Firefly analog of "modify" is **adapt the documented container
config recipe to the user's environment**:

1. **Start from the documented recipe.** Identify the public guide's
   recipe that matches the user's PTP role and profile. Quote it;
   do not author a new one from scratch.
2. **Diff against the user's environment.** Note the specific
   substitutions the user must make: interface name, domain number,
   transport choice, upstream master address (for slave / boundary
   roles), config file path, container image tag (always pulled
   from NGC per the public guide).
3. **Apply minimum-change.** Change only what the user's
   environment forces. Every additional deviation from the
   documented recipe widens the surface for an unintended PTP
   mismatch the operator will have to debug later.
4. **Re-validate against the four-axis table.** Each substitution
   is a chance to accidentally break one of the four PTP axes
   (role / profile / domain / interface). Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   one row at a time after every substitution.
5. **Re-validate against the END-TO-END discipline.** Per the
   END-TO-END rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   any change that affects which PHC device the host reads, or
   which port carries PTP, also affects the host-side follower's
   config. Update both together.

The agent's anti-pattern alert: a *"start from a generic `ptp4l`
config and adapt"* is almost always slower than starting from the
public Firefly Service Guide's recipe, because the Firefly config
schema is documented per the public guide and is not 1:1 with
upstream `linuxptp`.

## run

Bringing up the Firefly container and confirming PTP advances
through its state machine, with the host follower attached, BEFORE
layering any consumer workload on top.

1. **Pull the Firefly container image from NGC** at the tag the
   public Firefly Service Guide names for the operator's DOCA
   release. Quote the tag from the live guide; do NOT memorize or
   invent the tag.
2. **Start the container per the public Container Deployment Guide
   pattern.** Mount the Firefly config file at the path the public
   Firefly Service Guide names. The runtime command shape (e.g.
   `docker run` / `crictl` / BlueField container manager) is
   documented in the Container Deployment Guide reachable through
   [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
3. **Confirm the container is running, not restart-looping.** A
   restart loop is a layer-1 symptom per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   (container runtime / image tag / config mount); diagnose it
   before touching PTP config.
4. **Watch the Firefly container's logs for the PTP state-machine
   transitions.** The container's stdout is the primary
   observability surface. The PTP state should advance from
   `LISTENING` through `UNCALIBRATED` and into `SLAVE` (for slave
   role) or `MASTER` (for master role) within the timeline the
   public guide describes for the chosen profile.
5. **Confirm the BlueField PHC is being disciplined.** The PHC
   offset (relative to its master) and frequency adjustment are
   the numeric proof. Use the upstream PTP tooling (e.g. `pmc`,
   `phc_ctl`) the public guide cites; do not invent commands.
6. **Wire the host-side follower NOW.** Per the host-follower step
   in `## configure`, start chrony with the PHC source or `ptp4l` /
   `phc2sys` reading the PHC. Confirm `chronyc tracking` (or `pmc`
   against the host's PTP daemon) reports the BlueField PHC as the
   reference and a tight offset. Without this step the host clock
   does NOT follow the PHC.
7. **Single-event smoke (next: `## test` step 1).** Before driving
   any consumer workload, walk `## test` step 1 once to confirm
   end-to-end discipline holds; only then layer the consumer
   workload on top.

For the runtime version + container-tag cross-checks that underlie
*"my Firefly behaves differently from what the docs say"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run)
and apply the container-tag-lags-host-package overlay from
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

## test

Firefly has no "compile and unit-test" workflow — testing is
operational and end-to-end.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (role change, profile change, domain change, interface
change, transport change, host-follower change) re-opens the smoke
sweep. Skipping the re-run after a mutation is the failure mode this
loop replaces.

The eval-loop overlay (rows apply to every Firefly deployment, not
just one PTP profile):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 4 → 1 | Step 4 (consumer-workload check) often reveals an as-deployed gap in the host-follower or in the Firefly config; loop back to step 1 | [`## test`](#test) step 4 |
| 2 → ## debug | When the four-axis smoke does NOT progress past `LISTENING`, the deployment is non-functional — escalate to `## debug` layer 2 immediately, do not run later steps | [`## debug`](#debug) |
| 3 → ## configure → 3 | When the host follower does not track the PHC, the host-side config is wrong — loop back to `## configure` step 4 and re-run | [`## configure`](#configure) |
| 1..5 → ## run | Each loop iteration ends with a smoke; if all five pass, hand off to live `## run` traffic | [`## run`](#run) |

The agent's rule: every mutation re-opens the sweep. A configuration
change followed by *"it probably still works"* is exactly the
failure mode the iterative loop is here to prevent.

1. **End-to-end smoke.** With Firefly running and the host follower
   wired, confirm in this order: (a) Firefly container stdout shows
   PTP advanced to `SLAVE` or `MASTER` per the chosen role; (b)
   PHC offset reported by `pmc` / `phc_ctl` is within the
   profile's expected range; (c) host-side follower reports the
   PHC as its reference with a tight offset; (d) the host's
   `date` / `clock_gettime(CLOCK_REALTIME)` agrees with an
   independent source within the profile's spec.
2. **Four-axis smoke.** Confirm the negative case: temporarily
   misconfigure the domain number (e.g. set it to one the upstream
   master is NOT on) and confirm Firefly fails to peer. This
   validates the operator's understanding of the four-axis rule
   AND that the upstream master is in fact on the originally
   configured domain. Restore the configured domain afterwards.
3. **Host-follower smoke.** Stop the host-side follower and confirm
   the host clock starts to drift relative to the PHC; restart it
   and confirm the drift closes. This validates that the follower
   is in fact the thing disciplining the host clock and not, e.g.,
   a stray `chronyd` reading an NTP source.
4. **Consumer-workload smoke.** For Rivermax SMPTE, run the
   `doca-rmax` single-frame smoke (per
   [`doca-rmax TASKS.md ## test`](../../libs/doca-rmax/TASKS.md#test))
   with the disciplined PHC available; for 5G UPF / finance /
   distributed databases, the workload-side smoke is the workload's
   own concern. The Firefly contribution to the consumer smoke is
   *"the disciplined time was actually available when the workload
   asked for it"* — `chronyc tracking` snapshot at the time of the
   smoke is the evidence.
5. **Capability snapshot.** Save the *as-deployed* answer to:
   which Firefly container tag is running, which role / profile /
   domain / interface / transport are in effect, which follower
   mechanism the host is using, what the steady-state PHC offset
   looks like. This snapshot is the artifact that lets future
   debug sessions skip rediscovery.

## debug

Layered diagnosis. Walk the layers in this order; do not skip down
without clearing the layer above. The five layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

1. **Container runtime layer.** Is the Firefly container actually
   running and not restart-looping? Symptoms: container exits
   immediately, image pull fails, restart count climbing.
   Resolution: confirm the image tag matches what the public guide
   names for the operator's DOCA release; confirm the config mount
   path matches what the public guide names; confirm BlueField has
   the runtime configured per the public Container Deployment
   Guide. This layer is owned by the container runtime, not by
   PTP.
2. **Four-axis PTP-config layer.** Container green; PTP never
   advances past `LISTENING`, or no peers seen on the wire, or the
   wrong upstream master is selected. Resolution: walk the
   four-axis table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   one row at a time, comparing the Firefly config against what the
   upstream master / network actually presents. `tcpdump` for PTP
   traffic on the configured interface is the cheapest confirmation
   of which axis is wrong; defer to upstream Linux PTP / `tcpdump`
   docs for the exact filter.
3. **Host-follower layer.** PTP locks (Firefly says synced; PHC
   offset is tight); host OS clock drifts. Resolution: the
   host-side chrony / `ptp4l` / `phc2sys` is not configured to
   follow the PHC, OR it is configured but a competing NTP /
   chrony source is winning. Walk `chronyc tracking`,
   `chronyc sources`, and (if `ptp4l` is the follower) `pmc`
   against the host's PTP daemon. This layer is owned by upstream
   Linux PTP / chrony, not by Firefly — Firefly is correct.
4. **PTP-aware-path layer.** PTP locks; sync acquired; offset and
   jitter wildly past the profile's spec. Resolution: a non-PTP-
   aware switch in the path is silently adding variable latency
   that PTP cannot correct from the endpoint. Confirm via the
   network team / switch-side inspection. The fix is on the
   network side: insert boundary clocks at the non-PTP-aware
   switch, or replace the switch with a PTP-aware one. **Do NOT
   try to fix this from Firefly config; it is a network-path
   property, not a Firefly knob.**
5. **Consumer-workload layer.** Firefly correct; PHC correct; host
   clock follows PHC; but the consumer workload still reports time
   drift. Resolution: the workload is reading its own time source
   instead of the disciplined system clock. For Rivermax, walk
   [`doca-rmax TASKS.md ## debug`](../../libs/doca-rmax/TASKS.md#debug);
   for 5G UPF / finance / distributed DB, the workload's own docs.
6. **Version layer.** When the public Firefly Service Guide page
   appears to disagree with what the deployed container does, the
   docs version may not match the container tag. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 (partial install / version mismatch) and apply the
   container-tag overlay from
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
7. **Cross-cutting layer.** For env-side and program-side debug
   that is not Firefly-specific (host install, host kernel, DOCA
   library errors Firefly may surface), drop to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).

## Command appendix

Firefly-specific commands the verbs above reach for, grouped by
purpose so the agent picks the right family without searching prose.
Every row is a class — the agent must not invent flags beyond what
the row names; flag and command discovery is `--help` on the
installed tool or the public guide, not prose recall.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; the BlueField container manager's structured status output
   when available).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row.
   Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Container lifecycle | The BlueField container manager's start / stop / status command for the Firefly container, per the public Container Deployment Guide | [`## run`](#run) | Container `running`, restart count stable. |
| Container logs | The BlueField container manager's log-stream command for the Firefly container | [`## debug`](#debug) layer 1 + 2 | PTP state-machine transitions visible; no documented error / warning lines repeating. |
| PHC offset / frequency | `pmc -u -b 0 'GET CURRENT_DATA_SET'` or `phc_ctl /dev/ptpN get` (upstream Linux PTP tooling — quote the exact form from the public Firefly guide) | [`## run`](#run) step 5; [`## debug`](#debug) layer 2 | Offset within the chosen profile's spec; frequency adjustment converged. |
| Ports-state inspection | `pmc -u -b 0 'GET PORT_DATA_SET'` (upstream Linux PTP) | [`## debug`](#debug) layer 2 | Port state advanced past `LISTENING` to `SLAVE` / `MASTER` per role. |
| On-the-wire confirmation | `tcpdump` for PTP traffic on the configured interface (upstream Linux / `tcpdump` — defer to its own docs for the exact filter) | [`## debug`](#debug) layer 2 | PTP frames egress / ingress as the role expects; domain numbers in the frames match the configured domain. |
| Host-follower status (chrony) | `chronyc tracking` and `chronyc sources` (upstream chrony) | [`## run`](#run) step 6; [`## debug`](#debug) layer 3 | Reference ID corresponds to the PHC; offset tight; stratum / leap status sane. |
| Host-follower status (`ptp4l`) | `pmc` against the host-side PTP daemon (upstream Linux PTP) | [`## run`](#run) step 6; [`## debug`](#debug) layer 3 | Host's PTP daemon reports the PHC as its source; offset tight. |
| Container tag in use | The BlueField container manager's image-inspect command for the running Firefly container | [`## run`](#run) step 1; [`## debug`](#debug) layer 6 | Tag matches what the public Firefly Service Guide names for the operator's DOCA release. |

Three cross-cutting rules for this appendix:

- **Never invent a Firefly config key, container tag, or PTP
  command.** The public Firefly Service Guide is the contract;
  upstream Linux PTP / chrony / `tcpdump` docs are the secondary
  source for the cross-cutting PTP tools the guide reuses. Prose-
  derived flags are the most common hallucination failure for this
  skill.
- **Container before PTP.** When triaging, confirm the container
  layer (running, not restart-looping, image tag correct) before
  reading any PTP-layer command. A non-running container makes
  every PTP-layer command meaningless.
- **Cross-link instead of duplicate.** Cross-cutting env commands
  (port-state, `devlink`, `ip link`, `ethtool`) live in
  [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  this appendix names only the Firefly-specific ones.

## Deferred task verbs

- **Installing DOCA on the BlueField** — out of scope here. Route
  to [`doca-setup ## configure`](../../doca-setup/TASKS.md#configure)
  for env preparation and
  [`doca-setup ## test`](../../doca-setup/TASKS.md#test) for
  install health verification, or
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path.
- **Installing or configuring the host-side time-sync follower**
  (chrony / `ptp4l` / `phc2sys`) — out of scope here. The Firefly
  contract is *that* the host must be configured to follow the
  PHC; the follower's own config body is upstream Linux PTP /
  chrony documentation, not a Firefly concern.
- **Designing the PTP topology** (where to place boundary clocks,
  which switches must be PTP-aware, redundant masters, BMCA
  configuration) — out of scope here. That is a network-side /
  topology-side concern that the operator owns; Firefly only
  consumes the topology decision.
- **Building a custom DOCA application that reads the disciplined
  PHC** — not a Firefly question. Route to
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the canonical build pattern and the matching `libs/<library>`
  skill (e.g.
  [`doca-rmax ## configure`](../../libs/doca-rmax/TASKS.md#configure)
  for SMPTE) for the API surface.
- **Other DOCA services** (DMS, DTS, BlueMan, Flow-Inspector,
  HBN, Argus, …) — not Firefly. Route to
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services)
  for the routing table and the matching `services/<service>` skill
  when it exists (e.g.
  [`doca-dms ## configure`](../doca-dms/TASKS.md#configure) for
  device management). The container-shaped deployment pattern is
  shared; the per-service domain is different.

## Cross-cutting

- The public DOCA Firefly Service Guide is the single source of
  truth. Any config key, profile name, role name, container tag,
  or observability output the agent quotes must come from there,
  not from generic `ptp4l` / `linuxptp` / chrony knowledge.
- PTP is END-TO-END. The Firefly container disciplines the
  BlueField PHC; the host-side follower disciplines the host
  clock from the PHC; the consumer workload reads the disciplined
  time. All three legs are mandatory; naming only one is how
  end-to-end discipline silently breaks.
- Path-selection is mandatory up front. Firefly is the wrong
  answer when chrony / NTP suffices, when no PTP-aware path
  exists, or when pure software-side time precision is enough.
- Smoke before scale. Every consumer workload (Rivermax, 5G UPF,
  finance, distributed DB) goes after the end-to-end smoke
  passes, never before.
- For URL routing to the Firefly guide and other public DOCA
  documentation, see
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).
