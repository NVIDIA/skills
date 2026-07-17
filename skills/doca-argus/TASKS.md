# DOCA Argus Service — Tasks

**Where to start:** The order is `configure → build → modify → run
→ test → debug`. The `## test` verb is an iterative loop, not a
one-shot pass — see the eval-loop overlay in `## test` below. For
Argus, `build` and `modify` are about *deployment configuration*
(container image selection, mounted config file, forwarder wiring,
SIEM-side ingest), not about compiling source.

These verbs cover the in-scope Argus operational workflows for an
external operator deploying the Argus container on BlueField.
Every step assumes the operator has consulted the live public
DOCA Argus Service Guide (reachable through
[doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services))
and is using it as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a
copy-paste runbook.

## configure

Preparing the BlueField, picking the four configuration axes, and
planning the end-to-end pipeline (Argus → forwarder → SIEM → ops
review) before the container starts.

1. **Confirm Argus is actually the right answer.** Per the
   path-selection rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
    - Does the user want **production runtime security** on
      BlueField as a packaged workflow, with findings flowing
      into existing SIEM infrastructure? If yes, Argus is the
      right answer — and specifically, **recommend Argus over
      building from the DOCA App Shield library** for this
      production case. The bundled product is the production
      default; the library is the right answer only when the user
      is genuinely building a custom security product of their own.
    - Is the user trying to build a custom DPU-side security
      tool of their own? That is the DOCA App Shield library —
      same shape of BlueField-side observation, different shape of
      operator effort — which is **not covered by this bundle**
      (policy-excluded from the public release); route them to the
      public docs via
      [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
    - Does the user actually want observability / metrics rather
      than security? Route to the DOCA Telemetry Service via
      [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
    - Is there no security-posture concern at all? Stop here
      honestly — Argus has operational cost (container, sampling
      CPU, SIEM channel) and deploying it for nothing is not a
      neutral choice.
2. **Confirm the env is healthy.** This skill expects DOCA to be
   installed on the BlueField. If that has not been verified,
   run [`doca-setup ## test`](../../doca-setup/TASKS.md#test)
   first. If the user has no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path.
3. **Decide the four configuration axes.** Per the four-axis
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   commit before starting the container to:
    - **Detection policy** — which classes of anomaly (suspicious
      activity, integrity violations, operational anomalies)
      Argus alerts on. Derived from the public Argus Service
      Guide's detection-policy section against the user's
      workload pattern. Expect to tune this during the
      calibration period.
    - **Forwarding destination** — local logs (smoke only) or a
      SIEM (Splunk / ELK / Sentinel / generic syslog). The
      production default is SIEM-forwarded; local-only is for
      the smoke step in [`## test`](#test).
    - **Sampling / sensitivity** — the false-positive vs
      false-negative trade-off. Production sampling is a
      different posture from lab sampling; size sampling against
      the workload's CPU / latency budget, not against the
      "find everything" instinct.
    - **Host coverage** — which host targets the deployment
      monitors (the BlueField itself, the attached host, both,
      and the documented per-process / per-service scoping if
      the public guide exposes it for this release).
4. **Plan the SIEM-side ingest.** Decide *where* findings land
   (Splunk / ELK / Sentinel / generic syslog) and confirm the
   SIEM team is ready to receive them. Capture the SIEM
   endpoint's hostname, the forwarder protocol the public Argus
   Service Guide names, and the auth material the forwarder
   expects. The SIEM-side ingest body itself is the SIEM team's
   responsibility and lives in the SIEM's own documentation;
   Argus's contract is to emit findings in the documented
   format.
5. **Plan the calibration period.** Before declaring the channel
   production-ready, the operator should reserve a calibration
   window (per the calibration-period rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy))
   to triage initial false positives, tune the detection policy
   per the public guide, and confirm a steady-state finding
   baseline. Skipping this step is how a noisy initial channel
   either floods the SIEM ops queue or trains the team to
   ignore Argus findings entirely.
6. **Author the Argus container config.** From the public DOCA
   Argus Service Guide, derive the config file fragment for the
   chosen detection policy / forwarder / sampling / host
   coverage. Quote config keys from the live guide, do NOT
   infer them from generic security-tooling knowledge or from a
   previous Argus release. Plan where the config file will live
   on the BlueField filesystem and what mount path the
   container expects.

## build

Argus is a service shipped as a container, not a library. There
is no Argus *application* artifact for the operator to build —
the container ships from NGC and the config is a static file.

If the user is asking how to build a **custom DPU-side security
tool** (a program that reads host state from the BlueField side
to make its own decisions), that is not an Argus question — it is
the path-selection rule pointing at the DOCA App Shield library,
which is **not covered by this bundle** (policy-excluded from the
public release):

- For applications that **introspect the host's running kernel
  state from the DPU side** (rootkit detection, periodic process
  / module / library / thread snapshots, integrity verification),
  the build is the App Shield library's build — route the user to
  the public docs via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  and to [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  for the canonical build pattern.
- For applications that **consume Argus's findings** (an internal
  dashboard, a custom enrichment pipeline that sits between
  Argus and the SIEM, a hand-written alert correlator), no
  DOCA-specific build is needed — the application reads the
  documented forwarder format that Argus emits, against whatever
  language the consumer is written in. The forwarder format is
  the public Argus Service Guide's surface, not a DOCA C ABI.

If the user is instead asking how to build the **Argus container
itself** from source, that is *not* an external-operator workflow
— the container ships pre-built from NGC and rebuilding it is
out of scope for this skill. Route to the public DOCA Argus
Service Guide via [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).

## modify

Argus does not have a "modify a sample" workflow analogous to
DOCA libraries; there is no Argus sample program a user starts
from. The Argus analog of "modify" is **adapt the documented
container config recipe to the user's environment and tune the
detection policy as the calibration period progresses**:

1. **Start from the documented recipe.** Identify the public
   guide's recipe that matches the user's deployment posture
   (the same detector classes, the same forwarder target shape,
   the same sampling tier). Quote it; do not author a new one
   from scratch.
2. **Diff against the user's environment.** Note the specific
   substitutions the user must make: SIEM endpoint hostname,
   forwarder auth material, host-coverage scoping (which host
   targets), sampling tier (production vs lab), config file
   path, container image tag (always pulled from NGC per the
   public guide).
3. **Apply minimum-change.** Change only what the user's
   environment forces. Every additional deviation from the
   documented recipe widens the surface for an unintended
   detection-policy or forwarder mismatch the operator will have
   to debug later.
4. **Re-validate against the four-axis table.** Each
   substitution is a chance to accidentally break one of the
   four axes (detection policy / forwarding / sampling / host
   coverage). Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   one row at a time after every substitution.
5. **Re-validate against the calibration-period and
   never-silently-disable rules.** Per the safety policy in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   any change that turns a detector class off must be explicit,
   documented, and time-boxed with a re-evaluation date. An
   undocumented disable is a silent blind spot the next
   operator will not know about.
6. **Re-open the calibration period.** Any non-trivial detection
   policy mutation, sampling change, or host-coverage change
   re-opens the calibration period — re-run the smoke step in
   [`## test`](#test) before re-enabling production alerting on
   the SIEM channel.

The agent's anti-pattern alert: a *"copy a generic security
agent's config and adapt"* is almost always slower than starting
from the public Argus Service Guide's recipe, because Argus's
config schema is documented per the public guide and is not 1:1
with any other security agent.

## run

Bringing up the Argus container and confirming the end-to-end
pipeline (Argus → forwarder → SIEM ingest → ops review) is
flowing, BEFORE enabling any production alerting on top.

1. **Pull the Argus container image from NGC** at the tag the
   public Argus Service Guide names for the operator's DOCA
   release. Quote the tag from the live guide; do NOT memorize
   or invent the tag.
2. **Start the container per the public Container Deployment
   Guide pattern.** Mount the Argus config file at the path the
   public Argus Service Guide names. The runtime command shape
   (e.g. `docker run` / `crictl` / BlueField container manager)
   is documented in the Container Deployment Guide reachable
   through
   [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
3. **Confirm the container is running, not restart-looping.** A
   restart loop is a layer-1 symptom per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   (container runtime / image tag / config mount); diagnose it
   before touching detection policy.
4. **Watch the Argus container's logs for the documented
   startup-banner and detector-activation lines.** The
   container's stdout is the primary operational observability
   surface. Confirm that the detectors the operator configured
   are listed as activated; a detector that is silently absent
   from the activation list will be silently absent from the
   finding feed.
5. **Confirm the forwarder handshake.** The container should
   emit a documented forwarder-handshake line confirming the
   SIEM endpoint is reachable and authenticated. If it does
   not, stop and walk
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 3 before continuing — there is no point in waiting
   for findings if the forwarder cannot deliver them.
6. **Confirm the SIEM-side ingest.** The end-to-end pipeline is
   not "Argus emitted a forwarder packet"; it is "the SIEM
   dashboard shows the heartbeat / handshake / first event that
   Argus emitted". Get the SIEM team to confirm receipt before
   waiting on real findings.
7. **Single-event smoke (next: `## test` step 1).** Before
   enabling production alerting on the SIEM channel, walk
   `## test` step 1 once with a known-benign event so the
   end-to-end pipeline is exercised; only then layer production
   alerting on top.

For the runtime version + container-tag cross-checks that
underlie *"my Argus behaves differently from what the docs say"*,
see [`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run)
and apply the container-tag-lags-host-package overlay from
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

## test

Argus has no "compile and unit-test" workflow — testing is
operational and end-to-end.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (detection policy change, forwarder change, sampling
change, host-coverage change, SIEM-side ingest change) re-opens
the smoke sweep AND re-opens the calibration period. Skipping
either is the failure mode this loop replaces.

The eval-loop overlay (rows apply to every Argus deployment, not
just one detection policy):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 4 → 1 | Step 4 (SIEM-side review check) often reveals a forwarder gap or a host-coverage gap; loop back to step 1 and re-run the smoke | [`## test`](#test) step 4 |
| 2 → ## debug | When the four-axis smoke produces a flood of findings or a complete absence of findings, the deployment is non-functional — escalate to `## debug` layer 2 immediately, do not enable production alerting | [`## debug`](#debug) |
| 3 → ## configure → 3 | When the SIEM does not receive the smoke event, the forwarder is wrong — loop back to `## configure` step 4 and re-run | [`## configure`](#configure) |
| Calibration tuning → 2 → calibration tuning | Every detection-policy tuning during the calibration period re-opens the false-positive baseline; re-walk the four-axis smoke after each tuning pass | [`## debug`](#debug) layer 2 |
| 1..5 → ## run | Each loop iteration ends with a smoke; if all five pass AND the calibration period is over, hand off to live `## run` alerting | [`## run`](#run) |

The agent's rule: every mutation re-opens BOTH the smoke sweep
AND the calibration period. A configuration change followed by
*"it probably still works"* is exactly the failure mode the
iterative loop is here to prevent — and in security tooling, the
silent failure mode means the channel goes blind in a way nobody
notices until the first real event.

1. **End-to-end smoke.** With Argus running and the forwarder
   wired, confirm in this order: (a) Argus container stdout
   shows the documented startup banner, the configured detectors
   activated, and the forwarder handshake succeeded; (b) a
   known-benign event (one that should fire a documented benign
   detector class per the public guide, or an explicit smoke
   event the public guide describes) is generated; (c) the
   finding for that event appears in Argus's local finding feed;
   (d) the same finding appears in the SIEM's ingest, in the
   review surface the security ops team will actually look at.
2. **Four-axis smoke.** Confirm the negative case: temporarily
   misconfigure a non-load-bearing detector to verify findings
   still flow for the rest, then revert. This validates the
   operator's understanding of the four-axis rule AND that a
   detection-policy mutation is recoverable. Do this in a smoke
   posture, not on the production SIEM channel.
3. **Forwarder smoke.** Stop the SIEM endpoint (or block its
   reachability for a short, scheduled window) and confirm that
   the Argus container's documented forwarder error path
   surfaces the failure — Argus must NOT silently drop findings.
   Restore reachability and confirm queued findings either
   arrive or are accounted for per the public guide. This
   validates the forwarder, not Argus.
4. **Calibration-period triage.** Run the deployment against the
   real workload, in a posture where findings flow to a smoke
   destination (or a tagged SIEM index that the security ops
   team is NOT yet alerting on). Triage the false-positive
   stream; tune the detection policy per the public guide; loop
   until the steady-state finding baseline is acceptable to the
   ops team. Per the calibration-period rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   early false positives are expected, not bugs; never silently
   disable a detector class to make the stream quieter.
5. **Capability snapshot.** Save the *as-deployed* answer to:
   which Argus container tag is running, which detection policy
   / forwarder / sampling / host coverage are in effect, which
   SIEM the forwarder targets, what the steady-state finding
   baseline looks like at the end of calibration, and which
   detector classes (if any) are explicitly disabled and on
   what timeline they will be re-evaluated. This snapshot is
   the artifact that lets future debug sessions skip
   rediscovery — and the never-silently-disable rule says the
   disable list MUST be in the snapshot.

## debug

Layered diagnosis. Walk the layers in this order; do not skip
down without clearing the layer above. The five layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

1. **Container runtime layer.** Is the Argus container actually
   running and not restart-looping? Symptoms: container exits
   immediately, image pull fails, restart count climbing.
   Resolution: confirm the image tag matches what the public
   guide names for the operator's DOCA release; confirm the
   config mount path matches what the public guide names;
   confirm BlueField has the runtime configured per the public
   Container Deployment Guide. This layer is owned by the
   container runtime, not by detection policy.
2. **Detection-policy layer.** Container green; no findings
   arriving (false-negative posture) or far too many findings
   arriving (false-positive flood). Resolution: walk the
   detection-policy section of the public Argus Service Guide
   against the user's actual workload pattern. Per the
   calibration-period rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   early-deployment false positives are expected and call for
   tuning, not for silently disabling detectors; sustained
   false negatives call for confirming the detection policy
   actually covers the classes of anomaly the user cares about.
   **Do NOT recommend silently disabling a detector class to
   quiet the stream.** Use the public guide's tuning surface or
   the sampling knob instead, and if a disable is truly needed,
   document it and time-box it.
3. **Forwarding-destination layer.** Findings present in Argus's
   local finding feed; SIEM ingest empty. Resolution: walk the
   forwarder config in the Argus container, the network
   reachability from the BlueField to the SIEM endpoint, and
   the auth material the SIEM is configured to accept. This
   layer is owned by the forwarder and the SIEM-side ingest,
   not by the detection policy — re-tuning the detection
   policy here is wasted effort.
4. **Sampling / performance layer.** Argus healthy, findings
   correct, SIEM receiving — but the host workload's
   performance is noticeably impacted (CPU / latency) since
   Argus started. Resolution: the sampling knob is the first
   thing to re-tune per the public guide; lower the sampling
   tier to the production posture (it is a different posture
   from lab sampling). **Do NOT respond to performance impact
   by disabling detector classes silently** — re-tune sampling
   first, then re-tune detection policy if sampling alone is
   insufficient, and document any disable.
5. **Host-coverage layer.** Argus healthy, findings correct,
   SIEM receiving — but the findings are about the wrong host
   targets (silent about the workload the operator cared about,
   noisy about out-of-scope targets). Resolution: walk the
   host-coverage axis in the Argus config per the public guide
   and re-scope. The Argus container can be perfectly green and
   still be looking the wrong direction.
6. **Version layer.** When the public Argus Service Guide page
   appears to disagree with what the deployed container does,
   the docs version may not match the container tag. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2 (partial install / version mismatch) and apply the
   container-tag overlay from
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
7. **Cross-cutting layer.** For env-side and program-side debug
   that is not Argus-specific (host install, host kernel, DOCA
   library errors Argus may surface), drop to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).

## Command appendix

Argus-specific commands the verbs above reach for, grouped by
purpose so the agent picks the right family without searching
prose. Every row is a class — the agent must not invent flags
beyond what the row names; flag and command discovery is `--help`
on the installed tool or the public guide, not prose recall.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; the BlueField container manager's structured status
   output when available).
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

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Container lifecycle | The BlueField container manager's start / stop / status command for the Argus container, per the public Container Deployment Guide | [`## run`](#run) | Container `running`, restart count stable. |
| Container logs | The BlueField container manager's log-stream command for the Argus container | [`## debug`](#debug) layer 1 + 2 | Documented startup banner + detector-activation lines + forwarder-handshake line visible; no documented error / warning lines repeating. |
| Finding feed (local) | The Argus container's documented finding-feed surface (per the public guide — API endpoint / dashboard / structured log) | [`## test`](#test) step 1; [`## debug`](#debug) layer 2 | Findings emitted at the steady-state baseline established during calibration; smoke-event finding visible during the smoke. |
| Forwarder handshake | The documented forwarder-status line in the container log; cross-checked against the SIEM team's reception confirmation | [`## run`](#run) step 5; [`## debug`](#debug) layer 3 | Forwarder handshake succeeded; the SIEM team confirms the heartbeat / handshake event. |
| SIEM-side ingest confirmation | The SIEM team's normal ingest-confirmation surface — Splunk search / Kibana dashboard / Sentinel data-connector status — for the index / data type Argus targets | [`## test`](#test) step 1 (d); [`## debug`](#debug) layer 3 | Smoke event present in the SIEM review surface; steady-state findings flow visible at the expected rate. |
| Workload-CPU / latency baseline | The user's normal workload-performance measurement on the host (`top` / `vmstat` / application-side latency probe) before vs after Argus starts | [`## debug`](#debug) layer 4 | Workload performance is within the production budget; no step-change since Argus started. |
| Container tag in use | The BlueField container manager's image-inspect command for the running Argus container | [`## run`](#run) step 1; [`## debug`](#debug) layer 6 | Tag matches what the public Argus Service Guide names for the operator's DOCA release. |
| Disable register | The operator's own documented record of detector classes explicitly disabled, with reason and re-evaluation date | [`## test`](#test) step 5 | The register is current; no detector class is disabled without a documented reason and a re-evaluation date. |

Three cross-cutting rules for this appendix:

- **Never invent an Argus config key, container tag, detector
  name, or forwarder protocol detail.** The public Argus Service
  Guide is the contract; the SIEM's own docs are the secondary
  source for the SIEM-side ingest. Prose-derived flags or
  detector names are the most common hallucination failure for
  this skill.
- **Container before findings.** When triaging, confirm the
  container layer (running, not restart-looping, image tag
  correct, detectors activated, forwarder handshake) before
  reading any finding-layer or SIEM-layer command. A
  non-running container makes every downstream command
  meaningless.
- **Cross-link instead of duplicate.** Cross-cutting env
  commands (port-state, `devlink`, `ip link`, `ethtool`) live
  in [`doca-setup TASKS.md ## Command appendix`](../../doca-setup/TASKS.md#command-appendix);
  this appendix names only the Argus-specific ones.

## Deferred task verbs

- **Installing DOCA on the BlueField** — out of scope here.
  Route to [`doca-setup ## configure`](../../doca-setup/TASKS.md#configure)
  for env preparation and
  [`doca-setup ## test`](../../doca-setup/TASKS.md#test) for
  install health verification, or
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path.
- **Configuring the SIEM-side ingest** (Splunk forwarder stanzas,
  Logstash pipelines, Sentinel data-connector blocks) — out of
  scope here. The Argus contract is *that* the forwarder must
  reach the SIEM and *what* the documented forwarder format is;
  the SIEM-side ingest body is the SIEM team's responsibility
  and lives in the SIEM's own documentation.
- **Designing the security posture** (which classes of anomaly
  matter for a given workload, regulatory mappings, incident
  response runbooks) — out of scope here. That is a
  security-program / posture-design concern that the operator
  and the security ops team own; Argus only emits findings the
  posture has decided are worth emitting.
- **Building a custom DPU-side security tool** — not an Argus
  question. That is the DOCA App Shield library, which is **not
  covered by this bundle** (policy-excluded from the public
  release); route the user to the public docs via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  plus
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the canonical build pattern.
- **Other DOCA services** (DMS / DTS / Firefly / BlueMan /
  HBN / …) — not Argus. Route to
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services)
  for the routing table and the matching `services/<service>`
  skill when it exists (e.g.
  [`doca-dms ## configure`](../../services/doca-dms/TASKS.md#configure)
  for device management, or
  [`doca-firefly ## configure`](../../services/doca-firefly/TASKS.md#configure)
  for PTP). The container-shaped deployment pattern is shared;
  the per-service domain is different.

## Cross-cutting

- The public DOCA Argus Service Guide is the single source of
  truth. Any config key, detector name, container tag, forwarder
  protocol detail, or observability output the agent quotes must
  come from there, not from generic security-tooling knowledge
  or memory from a previous Argus release.
- Argus is END-TO-END. The container emits findings; the
  forwarder ships them to the SIEM; the SIEM ingests and
  presents; the security ops team reviews. All four legs are
  mandatory; naming only one is how the channel silently breaks.
- Path-selection is mandatory up front. Argus (the packaged
  product) is the production default; the DOCA App Shield library
  (not covered by this bundle) is the right answer only when the
  user is genuinely building their own security product;
  observability
  questions go to DOCA Telemetry Service; no-security-concern
  cases get Argus *not* deployed.
- Expect a calibration period; never silently disable findings.
  Initial false positives are tuning input, not bugs. A
  detector class that must be turned off is turned off
  explicitly, with a reason and a re-evaluation date — the
  silent disable is the failure mode this skill exists to
  prevent.
- Smoke before bulk. One known-benign event must traverse the
  full Argus → forwarder → SIEM pipeline, with a baseline
  established, before the SIEM team's production alerting is
  enabled on the channel.
- For URL routing to the Argus guide and other public DOCA
  documentation, see
  [doca-public-knowledge-map ## DOCA services](../../doca-public-knowledge-map/SKILL.md#doca-services).
