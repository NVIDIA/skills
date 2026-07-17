# DOCA Argus Service — Capabilities

**Where to start:** The pattern overview below names the recurring
Argus-class operational patterns. Pick the pattern first, then
drill into the H2 that owns the substance. For the *how* of
executing each pattern, jump to [TASKS.md](TASKS.md).

This file enumerates Argus's documented capabilities, deployment
shape, configuration axes, and operational behaviors as described
in the public DOCA Argus Service Guide. Treat it as a *map of what
is documented*, not a substitute for reading the live page when
configuring a real deployment. For the public URL itself, route
through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
— this skill does not duplicate the URL routing.

## Pattern overview

Every Argus-class question this skill teaches resolves into one of
FIVE patterns. The patterns are CLASSES — they apply across every
Argus deployment, not just one detection policy or one SIEM
consumer.

| Argus pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide Argus (packaged) vs App Shield library vs nothing | Production security workflow as a packaged product → Argus; custom DPU-side security tooling → the DOCA App Shield library (not covered by this bundle — route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)); no security posture concern → neither | [`## Safety policy`](#safety-policy) path-selection rule |
| 2. Pick the four configuration axes | Detection policy + forwarding destination + sampling / sensitivity + host coverage — every axis is a deployment hazard if wrong | [`## Capabilities and modes`](#capabilities-and-modes) four-axis table |
| 3. Wire the END-TO-END security pipeline | Argus container emits findings; the forwarder ships them to the SIEM (Splunk / ELK / …); the security ops team reviews; the Argus side and the SIEM side are independent moving parts | [`## Safety policy`](#safety-policy) END-TO-END rule + [`## Capabilities and modes`](#capabilities-and-modes) deployment shape |
| 4. Pair with a SIEM consumer | Splunk / ELK / Sentinel / generic syslog — each pairs with Argus in the same shape (Argus = finding emitter; SIEM = finding consumer) | [`## Capabilities and modes`](#capabilities-and-modes) pairing table |
| 5. Map an Argus symptom back to its layer | Container-runtime vs detection-policy vs forwarding vs sampling-performance vs host-coverage — five independent layers, each with its own owner | [`## Error taxonomy`](#error-taxonomy) layered split |

Two cross-cutting rules that apply to *every* pattern above:

- **Security tooling is sensitive — never silently disable
  findings.** The most common operator failure for Argus is to
  *quietly turn off a noisy finding class* during the calibration
  period and then forget that the channel is now blind to that
  class. The honest moves are: (a) tune the detection policy with
  the public guide so the noisy class is no longer over-firing,
  (b) raise the sampling threshold so the class is observed less
  aggressively, or (c) explicitly document and time-box the
  disable. Silent disables are the failure mode this skill exists
  to prevent.
- **Operate the documented path; do not invent one.** Argus's
  detection-policy schema, container image source, forwarder
  formats, sampling knobs, and observability surface are all
  documented in the public DOCA Argus Service Guide. Quoting
  config keys, image tags, detection-rule names, or CLI flags not
  in the public guide is the most common hallucination failure
  mode for this skill.

## Capabilities and modes

### Service shape

Argus is a **long-running container** that ships from NGC and
runs on the BlueField Arm cores. The container is the daemon: it
owns the runtime-security observation surface (monitoring BlueField
itself and the host it is attached to for suspicious activity,
integrity violations, and operational anomalies), it owns the
finding emission, and it owns the documented forwarder. There is
no host-side Argus binary the user installs — Argus is the
container; the host's relationship to Argus is that Argus
*observes* the host, not that Argus *runs on* the host.

Three architectural properties the operator must hold throughout:

- **The container is the unit of deployment.** Operators do not
  start `argus` as a host binary; they start the Argus container
  per the public Container Deployment Guide pattern (same shape
  as every other DOCA service container — see the sibling
  [`doca-dms`](../doca-dms/SKILL.md) and
  [`doca-firefly`](../doca-firefly/SKILL.md) for the same shape
  on different per-service domains).
- **Argus is a packaged product, not a library.** The whole point
  of Argus is that the detection logic, the finding format, and
  the forwarder integration ship as one operationally-ready unit.
  An operator who finds themselves writing their own detection
  loop has reached for the wrong artifact and should be routed to
  the DOCA App Shield library — the lower-level library that custom
  security tooling builds on, which is **not covered by this
  bundle** (policy-excluded from the public release; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
- **Findings are the primary output, not metrics.** Argus's
  surface is "did something security-relevant happen on this
  BlueField + host pair, and what is it" — not "what is the CPU
  utilization right now". An operator looking for general
  observability / metrics is asking the wrong service; route
  them to the DOCA Telemetry Service via
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Deployment shape

The public DOCA Argus Service Guide documents the container
deployment on BlueField Arm. The shape lines up with every other
DOCA service container — pull from NGC, mount the config, start
under the documented runtime (the BlueField OS's container manager
per the public Container Deployment Guide). For the canonical
container-deployment recipe shared with the other DOCA service
containers, route through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

Two deployment-shape rules:

- **BlueField Arm only.** Argus is a BlueField-side service; it
  does not run on the host. The host's relationship to Argus is
  via the observation surface that Argus reaches across (Argus
  observes the host from the DPU side, the same architectural
  shape as the DOCA App Shield library, only
  packaged) and via the network (forwarder traffic from the
  BlueField to the SIEM).
- **One Argus deployment per BlueField, one configured posture.**
  Argus's detection policy is the operator's posture decision for
  that BlueField + host pair. Running two Argus containers on the
  same BlueField with conflicting policies multiplies findings
  without multiplying signal; this is not a redundancy strategy.

### Four-axis configuration

Every Argus deployment must commit to four configuration axes
before starting the container. Get any one wrong and the
deployment fails in a different mode (no findings / wrong
findings / unforwarded findings / performance-impacted host). The
axes are jointly documented in the public Argus Service Guide;
quote the exact valid values from there rather than from memory.

| Axis | Class shape | Mismatch symptom | Where to look |
| --- | --- | --- | --- |
| **Detection policy** | Which classes of anomaly Argus alerts on — suspicious-activity classes, integrity-violation classes, operational-anomaly classes. Each class is a documented detector in the public guide | Too lenient → container green, no findings ever arrive even though the host workload is doing things; too strict → a flood of false-positive findings that overwhelms the SIEM channel and trains ops to ignore it | Public DOCA Argus Service Guide's detection-policy section |
| **Forwarding destination** | Where findings go — local logs (visible only via the container's log stream), or forwarded to a SIEM (Splunk / ELK / Sentinel / generic syslog) via the documented forwarder | Findings generated but the SIEM channel stays empty → the forwarder is misconfigured, the SIEM endpoint is unreachable from the BlueField, or the auth between forwarder and SIEM is mismatched | Public DOCA Argus Service Guide's forwarder section |
| **Sampling / sensitivity** | The false-positive vs false-negative trade-off knob — how aggressively Argus observes / samples the host. Higher = more findings (and more confidence, and more CPU); lower = fewer findings (and risk of missing real events, and less CPU) | Sampling too high for production → noticeable workload-CPU impact; sampling too low → genuine events get missed because Argus did not look closely enough | Public DOCA Argus Service Guide's sampling section |
| **Host coverage** | Which host targets the Argus deployment monitors — the BlueField itself, the attached host, both, and (if the deployment supports it per the public guide) which subsets of the host's processes / services are in scope | Wrong host coverage = systematic blind spot. The Argus container can be perfectly healthy and emitting findings *about a target the user does not care about* while staying silent about the target they do | Public DOCA Argus Service Guide's host-coverage section |

The agent's rule: **the four-axis decision precedes everything
else**. A deployment that starts the container before the operator
can name the detection policy, forwarding destination, sampling
target, and host coverage is going to debug the wrong axis first.
Force the decision up front.

### Pairing with SIEM consumers

Argus is the finding-emitting side of every supported pairing.
The SIEM is the finding-consuming side. Both sides must be wired;
Argus alone is not a finished deployment.

| SIEM consumer | Why it pairs with Argus | Pairing shape |
| --- | --- | --- |
| Splunk | Standard enterprise SIEM; many security ops teams already triage on Splunk | Argus's documented forwarder emits findings in the documented format; the SIEM team's Splunk side ingests via the SIEM's normal forwarder-receive path; review happens in Splunk dashboards / alerts |
| ELK (Elasticsearch + Logstash + Kibana) | Open-source SIEM stack; common for teams self-hosting | Same shape — Argus emits findings; Logstash / Beats / equivalent ingests on the SIEM side; review happens in Kibana dashboards / alerts |
| Microsoft Sentinel | Cloud-hosted SIEM; common when the security ops team is already on Azure | Same shape — Argus emits findings; the Sentinel data-connector path ingests; review happens in Sentinel |
| Generic syslog / file destination | Floor case when there is no SIEM in front of the deployment yet, or when the operator is just standing the channel up for the first time | Same shape — Argus emits findings in the documented forwarder format; the destination is a local file or a syslog receiver; this is the right shape for the smoke step in [`TASKS.md ## test`](TASKS.md#test) before wiring the production SIEM |

The agent's rule: when the user mentions a SIEM by name, name
Argus *and* the SIEM-side ingest *and* the security-ops review
step in the same breath. Naming only the Argus side is how the
end-to-end pipeline silently breaks: the channel is "up" from
Argus's perspective and "empty" from the SIEM's perspective, and
nobody notices until the first real event is missed.

### Configuration model

The Argus container is configured by a documented config file
that the operator mounts into the container at the path the
public guide names. The config file declares the four-axis
configuration (detection policy, forwarding destination, sampling,
host coverage) plus any advanced security knobs the user's
posture expects. Quote config keys from the live public Argus
Service Guide; do not infer them from generic security-tooling
knowledge — Argus's config schema is documented per the public
guide and is not 1:1 with any other security agent.

For deployments that need to evolve their detection policy over
time (which every long-lived Argus deployment will), the agent
should walk the operator through the public guide's documented
policy-evolution procedure rather than ad-hoc editing — every
mutation re-opens the calibration period per the
calibration-period rule in [`## Safety policy`](#safety-policy).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The Argus-specific overlay** is:

- **Argus is a NGC container; the container tag is the runtime
  version anchor.** Same pattern as
  [`doca-dms`](../doca-dms/SKILL.md) and
  [`doca-firefly`](../doca-firefly/SKILL.md): the Argus container
  ships from NGC with its own tag that may lag the host's DOCA
  package version, and the relevant version anchor for an
  as-deployed Argus is the container tag pulled, not
  `pkg-config --modversion` on the host. Always quote both
  versions when the user reports an Argus behavior; if they
  diverge, route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before diagnosing the Argus behavior itself.
- **Detection-policy schema is version-bound.** The detector
  classes Argus supports and the keys the config schema accepts
  evolve between releases. When the user asks *"does this
  detector / config key work on my deployment?"*, the
  authoritative answer is the public Argus Service Guide page
  whose version matches the container tag pulled — not memory
  from a previous release.
- **Read the public Argus Service Guide version header.** The
  guide is versioned; the on-page version must match the
  container tag the operator is using. A mismatch between the
  docs version and the container tag is the canonical *"my
  config doesn't work even though it matches the docs"* failure
  mode for Argus.

## Error taxonomy

Argus errors fall into five layers, each with its own owner. The
agent's rule: walk the layers in order; do NOT skip down without
clearing the layer above.

| Layer | Symptom | Root cause class | Where to fix |
| --- | --- | --- | --- |
| 1. Container runtime | Container fails to start, restart-loops, exits immediately, image pull fails | Image tag wrong, registry credentials missing, BlueField runtime not configured to run this container, config file mount path wrong | BlueField container runtime + the public Container Deployment Guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| 2. Detection policy | Container green, no findings ever arrive (false-negative posture); or container green, the SIEM channel is buried in findings (false-positive flood) | Detection policy too lenient for the workload pattern (nothing the user cares about will trip the configured detectors); or detection policy too strict for the workload pattern (every benign behavior trips a detector). This is the layer the calibration period in [`## Safety policy`](#safety-policy) exists to tune | The detection-policy config in the public Argus Service Guide — not the container, not the forwarder, not the sampling knob |
| 3. Forwarding destination | Findings are generated (visible in the container's local log / finding feed) but the SIEM channel stays empty | The forwarder is misconfigured (wrong endpoint, wrong protocol), the SIEM endpoint is unreachable from the BlueField (network reachability), or the auth between forwarder and SIEM is mismatched (wrong token / wrong cert) | The forwarder config in the Argus container + the SIEM-side ingest config — the network reachability and auth boundary between them is the most common cause |
| 4. Sampling / performance | Argus is healthy, finding emission is correct, the SIEM is receiving — but the host workload's performance is noticeably impacted (CPU / latency) since Argus started | Sampling rate too high for the production workload. Argus's observation surface costs something; production sampling is a different posture from lab sampling | The sampling knob in the Argus config per the public guide. Do NOT respond to performance impact by silently disabling detector classes — re-tune sampling first, then re-tune detection policy if needed |
| 5. Host coverage | Argus is healthy, finding emission is correct, the SIEM is receiving — but the findings are *about the wrong host targets* (Argus is silent about the workload the operator cared about, or noisy about a host target the operator considers out of scope) | Wrong host-coverage configuration. The Argus container can be perfectly green and still be looking the wrong direction | The host-coverage axis in the Argus config per the public guide |

The agent's rule: **never recommend a detection-policy change
without first identifying which of the five layers is the cause**.
The most common debug failure for this skill is misreading a
layer-3 symptom (findings not reaching SIEM) as a layer-2 problem
(detection policy) and rewriting the detection policy when the
fix is on the forwarder. Equally common: misreading a layer-4
symptom (performance impact) as "Argus is over-firing" and
silently disabling a detector class — see the safety policy.

## Observability

Documented observability surfaces the agent should reach for, in
order of how cheaply they answer the *"is Argus actually working"*
question:

1. **Container state.** First — is the Argus container actually
   running? The BlueField container manager reports container
   status, restart count, and the container's stdout / stderr
   log stream. A restart loop is a layer-1 (container runtime)
   symptom per [`## Error taxonomy`](#error-taxonomy); diagnose
   it before touching detection policy.
2. **Argus's own logs.** The container's stdout (and any
   documented log destination the public guide specifies) is the
   primary Argus operational observability surface. Look for the
   documented startup-banner lines, the documented detector
   activation lines, and the documented forwarder-handshake
   lines. The agent should NOT invent log line formats; quote
   what the live container is emitting.
3. **Finding feed.** Argus's finding output (whether via API,
   dashboard, or the documented forwarder format) is the proof
   that detectors are actually firing. A green container with a
   silent finding feed for 24h is a layer-2 (detection policy too
   lenient) symptom — or, if the SIEM is showing findings the
   local feed does not, a forwarder-loopback bug — not a healthy
   deployment.
4. **SIEM-side ingest confirmation.** The SIEM is where the
   security ops team actually reviews. The end-to-end smoke is
   not "Argus emitted a finding"; it is "the SIEM dashboard
   shows the finding that Argus emitted". The agent must teach
   the user to verify the SIEM-side ingest, not just the
   Argus-side emit.
5. **Workload performance baseline.** When the agent suspects
   layer 4 (sampling / performance), the cheapest confirmation
   is a workload-CPU / latency comparison before vs after Argus
   was started. If the workload was healthy without Argus and is
   degraded with Argus running, the sampling knob is the first
   thing to re-tune, not the detection policy.

For the cross-library debug-time observability (`DOCA_LOG_LEVEL`,
`--sdk-log-level`, the trace build flavor — relevant when Argus
calls into a DOCA library that emits structured logs), see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Argus's safety surface is **path-selection first**, then the
END-TO-END discipline, then the calibration-period rule, then
the never-silently-disable rule, then the smoke-before-bulk rule.
Argus is a security tool; the agent's safety bar is higher here
than for a non-security service, because the failure modes are
silent.

- **Path-selection rule (load-bearing).** Argus is the right
  answer only when the user wants **production runtime security
  on BlueField as a packaged workflow**. Concretely:
    - Use Argus when the operator needs runtime security on
      BlueField, wants a bundled detection + forwarding workflow,
      and is going to integrate with existing SIEM infrastructure
      (Splunk / ELK / Sentinel / …). This is the production
      default for most operators in this position.
    - **First recommend Argus (the packaged product) over the
      DOCA App Shield library for production security use cases.**
      A response that walks an external operator into building
      their own detection loop on App Shield as the default answer
      for production security is wrong by construction — the
      operator gets to
      own the detection logic, the rule tuning, the forwarder
      integration, and the lifetime of all of it, when Argus
      already ships those.
    - Do NOT reach for Argus when there is no security-posture
      concern (Argus has operational cost — container, sampling
      CPU, SIEM channel — for nothing); when the user actually
      wants observability / metrics rather than security (route
      to the DOCA Telemetry Service via
      [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md));
      or when the user is genuinely building a custom security
      product of their own that needs to ship its own decision
      logic (that is the DOCA App Shield library — same shape of
      BlueField-side observation, different shape of operator
      effort — which is not covered by this bundle; route to the
      public docs via
      [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
- **END-TO-END discipline (load-bearing).** Argus emits findings.
  The forwarder ships them to the SIEM. The SIEM ingests, stores,
  and presents them. The security ops team reviews. An Argus
  deployment that names only the BlueField side and stops there
  is a deployment that will *fail silently at the forwarder or
  SIEM*, not a deployment that works. The agent must always
  teach the four legs together: Argus container → forwarder →
  SIEM ingest → ops review.
- **Calibration-period rule (load-bearing).** Every Argus
  deployment owes its workload a calibration period — typically
  some initial false positives as the detection policy meets the
  real workload pattern for the first time, followed by tuning
  passes that drive false positives down without losing real
  detections. **The first wave of false positives is expected,
  not a bug.** An agent that diagnoses early false positives as
  *"Argus is broken"* or *"the detector class is wrong"* without
  first acknowledging the calibration period is misreading the
  layer. The fix is a policy tuning pass per the public guide,
  done while the deployment continues to run and emit findings —
  not a disable.
- **Never silently disable findings (load-bearing).** When a
  detector class is over-firing during the calibration period,
  the honest moves are: (a) re-tune the detection policy per the
  public guide so the class is no longer over-firing on this
  workload, (b) raise the sampling threshold so the class is
  observed less aggressively, or (c) explicitly document and
  time-box a disable with a re-evaluation date. **Silent
  disables are forbidden.** A disabled detector class is a known
  blind spot; an undocumented disable becomes an unknown blind
  spot the next time the team rotates, and the agent's only
  job in this corner is to keep the disable from becoming
  silent.
- **Smoke before bulk (load-bearing).** Before pointing the
  SIEM team's production review channel at the Argus deployment,
  the agent must walk the user through a smoke: Argus container
  running and not restart-looping, one known-benign event
  traverses Argus → forwarder → SIEM (proving the end-to-end
  pipeline works), and a baseline of expected steady-state
  findings is established. Only then enable the production
  alerting on top. A SIEM channel that goes from "no Argus" to
  "all production alerts on" without a smoke step silently uses
  a wrong baseline, and the bisection across Argus / forwarder /
  SIEM is much harder when the first real event arrives.
- **One Argus per BlueField, one posture.** Two Argus containers
  on the same BlueField with conflicting detection policies is a
  configuration error; the agent must NOT recommend it as a
  redundancy strategy. Security-side redundancy is a SIEM-side
  concern (the SIEM's own HA story) that does not require
  multiple Argus containers.

## Public-source pointer

The single canonical public source for Argus is the **DOCA Argus
Service Guide**, reachable through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
Verify that the version of the guide matches the Argus container
tag pulled on the BlueField — Argus's config surface, supported
detector classes, forwarder formats, and observability output are
documented to evolve, so config keys, detector names, and
forwarder protocol details can change between releases.
