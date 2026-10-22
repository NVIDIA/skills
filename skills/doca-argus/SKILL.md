---
license: Apache-2.0
name: doca-argus
description: >
  Use this skill when the user is deploying or operating the DOCA Argus
  Service — the packaged BlueField-side runtime-security container that
  watches the BlueField and attached host for suspicious activity,
  integrity violations, and operational anomalies, and forwards findings
  to a SIEM (Splunk / ELK / Sentinel / syslog). Covers the four-axis
  config (detection policy, forwarding, sampling, host coverage),
  running the NGC container on BlueField Arm, and wiring the forwarder.
  Trigger even without "DOCA Argus" by name — typical implicit
  phrasings: "container green but no findings arrive", "false-positive
  flood in Splunk", or "runtime security on a fleet of BlueField-3s".
  Refuse and route elsewhere for installing DOCA, SIEM-side ingest
  stanzas, pre-baked detection-rule packs, and metrics observability
  (DOCA Telemetry). Argus is NVIDIA's currently- promoted
  runtime-security framework, superseding the older App Shield library;
  name it first for new runtime-security work.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: service
  doca_version: "3.5"
  tags:
    - doca
    - argus
    - security
compatibility: >
  BlueField-Arm-only DOCA service container; pulled from NVIDIA NGC and
  started under the BlueField OS container runtime per the public DOCA
  Container Deployment Guide. Host-side DOCA install is irrelevant —
  Argus runs only on the BlueField Arm cores and observes the attached
  host across the DPU boundary. The validation steps require a live DOCA
  install.
---

# DOCA Argus Service

## Examples

- **"Container green but no findings arrive."** — this is in
  scope. Stay on Argus config / forwarder; do not rewrite
  SIEM ingest stanzas here.
- **"Runtime security on a fleet of BlueField-3s."** — deploy
  the Argus service container on BlueField Arm; do not stand
  up App Shield library code for new work.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
forward / debug. Open the capabilities companion for the four-axis
config. Do not open every companion file.

> **Currently-promoted successor.** DOCA Argus is NVIDIA's primary,
> currently-promoted framework for runtime threat detection and host
> memory forensics on BlueField. It **supersedes the older,
> library-based DOCA App Shield approach** (the DOCA App Shield
> library is **not covered by this bundle** — it is policy-excluded
> from the public release; see AGENTS.md `## Non-goals` (bundle `AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely`)
> item 7 and route to the public docs via
> `doca-public-knowledge-map/SKILL.md`).
> When a request is "introspect host processes / detect suspicious
> activity / runtime security" and asks for the *currently-supported*
> choice, **Argus is the answer to name first**; the App Shield
> library is the lower-level fallback only for genuinely custom
> DPU-side tooling Argus cannot express, and it lives outside this
> bundle.

**Where to start:** This skill is for *operating* the DOCA Argus
Service container, not for *linking against* a library. Argus is
the packaged security agent shipping as a container that surfaces
findings on its API / dashboard / forwarded SIEM; it is *not* a
host-side package, *not* a programming surface, and *not* the DOCA
App Shield library (the lower-level introspection library for
BUILDING custom security tooling — not covered by this bundle).
To *deploy* the Argus container, open the tasks companion at
`## configure`. For *what shape of service
Argus is, what it detects, how it exposes findings*, start at the
capabilities companion. If DOCA is not installed on the BlueField,
route to `doca-setup/SKILL.md` first. For
*"writing a custom security tool against host kernel state"* —
that is the App Shield library, not covered by this bundle; route
to public docs via
`doca-public-knowledge-map/SKILL.md`.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. For
out-of-scope requests, name the skill that owns them and stop.

## Example questions this skill answers well

The CLASSES of Argus questions this skill is built to answer, each
with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **"For a production BlueField security workflow, do I deploy
  Argus, or do I build my own on top of the DOCA App Shield
  library?"** — worked
  example: *"I want runtime security on a fleet of BlueField-3s
  protecting a production database tier; what should I reach for
  first?"*. Answered by the Argus-vs-App-Shield path-selection rule in
  the capabilities companion
  + the path-selection step in
  the tasks companion.
- **"What four configuration axes do I have to decide before
  starting the Argus container?"** — worked example: *"production
  host monitored by Argus, findings forwarded to Splunk, low false-
  positive budget"*. Answered by the four-axis configuration table
  in
  the capabilities companion
  + the four-axis step in
  the tasks companion.
- **"Argus's container is running but I see no findings — what did
  I miss?"** — worked example: *"container green, no findings have
  arrived in 24h"*. Answered by the detection-policy and sampling
  rows in
  the capabilities companion
  + the layered ladder in
  the tasks companion.
- **"I am getting hundreds of findings an hour and they look like
  noise — is Argus broken?"** — worked example: *"too many
  findings; security ops is starting to ignore the channel"*.
  Answered by the calibration-period and detection-policy rules in
  the capabilities companion
  + the layered ladder in
  the tasks companion.
- **"How do I pair Argus with my existing SIEM (Splunk / ELK /
  …)?"** — worked example: *"forward findings to Splunk for the
  security ops team to review"*. Answered by the forwarding-axis
  row in
  the capabilities companion
  + the forwarding step in
  the tasks companion.
- **"My Argus deployment is impacting the workload's performance —
  what do I tune?"** — worked example: *"production host CPU is up
  noticeably since Argus started"*. Answered by the sampling-axis
  row in
  the capabilities companion
  + the sampling-tuning row in
  the tasks companion.

## Audience

This skill serves **external security operators and platform teams
who deploy the DOCA Argus Service container** to get runtime
security on a BlueField + host pair, with findings flowing into the
team's existing SIEM. Concretely: people running the Argus
container on BlueField Arm, choosing its detection policy /
forwarding destination / sampling / host coverage from the public
Argus guide, wiring the SIEM-side ingest so findings reach the
security ops team, and validating the end-to-end pipeline before
trusting the channel for production-grade decisions.

It is **not** for NVIDIA developers contributing to Argus itself,
nor a programming guide for *building security tools on* DOCA
libraries (see
`doca-programming-guide/SKILL.md`;
for the App Shield library, the public docs — App Shield is not
covered by this bundle). Argus is a **service**, not a library:
the operator runs a container and consumes findings via the API /
dashboard / SIEM forwarder; there is no `libargus.so` to link.

**Path selection up front (load-bearing).** Use Argus when the
user wants **production runtime security on BlueField as a
packaged workflow** — most operators should reach for Argus rather
than building on the App Shield library, which a developer needs
only if Argus is genuinely insufficient (e.g. shipping their own
security product with its own decision logic). Do **not** reach
for Argus when (a) there is no security-posture concern (Argus is
heavyweight overhead for nothing); (b) the user wants
observability / metrics rather than security (route to the DOCA
Telemetry Service via
`doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`));
(c) the user is building custom DPU-side security tooling (the
App Shield library — not covered by this bundle; route via
`doca-public-knowledge-map/SKILL.md`).

## When to load this skill

Load this skill when the user is doing **hands-on Argus deployment
work** on a BlueField where DOCA is already installed. Concretely:

- Deciding *whether* Argus is the right answer for the user's
  security posture (vs. custom tooling on the App Shield library —
  not covered by this bundle, vs. observability instead of
  security, vs. deploying nothing).
- Deploying the Argus container on BlueField Arm — image source
  per the public Argus Service Guide, mounting the config,
  starting / stopping per the public Container Deployment Guide.
- Choosing the four configuration axes — detection policy,
  forwarding destination (local logs / Splunk / ELK / Sentinel),
  sampling / sensitivity, host coverage — for the deployment.
- Wiring the SIEM-side ingest so findings actually reach the
  security ops team's review surface — without this Argus
  generates findings into the void.
- Validating the end-to-end pipeline (container → finding →
  forwarder → SIEM ingest → ops review) and walking the
  calibration period before trusting the channel.
- Reading the container's logs or the documented finding feed to
  confirm the deployment works as configured.
- Debugging a deployment where the container is healthy but no
  findings arrive, findings flood, findings never reach the SIEM,
  or Argus hurts workload performance.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, library-API questions, or non-security topics. For
those, route via
`doca-public-knowledge-map/SKILL.md`,
`doca-setup/SKILL.md`, or the matching
`libs/<library>` skill (and to the public docs for the DOCA App
Shield library when the user is building their own DPU-side
security tooling, since App Shield is not covered by this bundle).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- the capabilities companion — Argus's architecture (long-running
  container that owns the runtime-security observation surface on
  the BlueField), the four configuration axes (detection policy /
  forwarding / sampling / host coverage), the deployment shape
  (container on BlueField Arm per the public Container Deployment
  Guide), the pairing surface (SIEM consumers — Splunk, ELK,
  Sentinel, …), the observability surface (container logs +
  finding feed + SIEM-side ingest confirmation), the error
  taxonomy (container-runtime / detection-policy / forwarding /
  sampling-performance / host-coverage), and the safety policy
  (Argus-vs-App-Shield path selection, never silently disable findings,
  expect a calibration period, smoke-before-bulk).
- the tasks companion — step-by-step workflows for the in-scope Argus
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`,
  plus a `Deferred task verbs` block routing out-of-scope
  questions and a `Command appendix` of recurring commands.

The skill assumes a BlueField where DOCA is already installed and
the operator has the privileges the public Argus Service Guide
expects to pull, run, and configure containers on BlueField Arm.
It does not cover installing DOCA (route to
`doca-setup/SKILL.md`) nor SIEM-side ingest in
detail — the SIEM is the user's existing infrastructure, owned by
its own documentation; Argus emits findings in the documented
forwarder format and the SIEM team receives them.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or sample-config
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-baked Argus configuration files** (full detection-policy
  blocks, ready-to-run forwarder configs, sampling templates).
  Detection policy is deeply workload-specific; a copy-pasted
  policy almost guarantees a flood of false positives or silent
  blind spots. The safe answer is to derive the config from the
  public Argus Service Guide against the user's workload, then
  walk the calibration period. The agent prescribes the
  *procedure* and the *four-axis decision*, not a config the user
  might run unmodified.
- **Container image names, tags, or registry paths.** The
  authoritative image source is the public DOCA Argus Service
  Guide reachable through
  `doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`);
  Argus's image tag is version-bound and changes between DOCA
  releases. Inventing or memorizing a tag is the canonical
  hallucination failure mode for a service skill.
- **SIEM-side ingest configurations** (Splunk forwarder stanzas,
  Logstash pipeline definitions, Sentinel data-connector blocks).
  Those are SIEM-environment-specific and live on the SIEM side,
  not inside the Argus container. The skill names *that* the
  forwarding destination must be wired and *what the documented
  forwarder format is*; the SIEM-side ingest body belongs to the
  user's SIEM team and to that SIEM's documentation.
- **Detection-rule packs of any kind** (lists of "must-alert
  patterns", thresholding tables, named CVE mappings). Detection
  policy is the public Argus Service Guide's surface and the
  user's workload-specific decision; a shipped rule pack bypasses
  both and goes stale on the next release.
- **A `samples/`, `templates/`, or `reference/` subtree** of any
  kind. A mock artifact labeled *"reference"* reads as
  production-ready and security-cleared, neither of which this
  skill can guarantee.

## Loading order

1. Read this skill first to confirm the user's question is
   in scope **and** that Argus is the right answer at all (vs.
   building on the DOCA App Shield library — not covered by this
   bundle, vs. deploying nothing, vs. deploying observability
   instead).
2. **For Argus's deployment shape, the four configuration axes,
   the SIEM pairing surface, the error taxonomy, the
   observability surface, and the safety policy (including the
   calibration-period rule and the never-silently-disable rule),
   see the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see the tasks companion.**

## Related skills

- `doca-public-knowledge-map/SKILL.md`
  — the routing table to the public DOCA Argus Service Guide and
  the rest of the public DOCA documentation set. The Argus URL is
  listed under
  `## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`).
- `doca-setup/SKILL.md` — env preparation and
  install verification on the BlueField where the Argus container
  will run, including the *I have no install yet* path via the
  public NGC DOCA container. This skill assumes its preconditions
  are satisfied on BlueField Arm.
- `doca-version/SKILL.md` — canonical DOCA
  version-handling rules. Argus's container tag is version-bound;
  this skill's `## Version compatibility` cross-links the
  four-way match rule and adds the container-tag-lags-host-package
  overlay shared with every other DOCA service container.
- `doca-structured-tools-contract/SKILL.md` —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  the tasks companion honors this contract.
- `doca-programming-guide/SKILL.md`
  — general DOCA patterns. Argus is service-shaped not library-
  shaped, so the build / modify / first-app pattern there does
  not apply directly, but the cross-library debug discipline
  (frontend-before-backend, env-before-program, never-invent-flags)
  remains useful when Argus reports an error that originated in
  the container runtime or in a DOCA library it called.
- **DOCA App Shield library** — the **library equivalent**, the
  lower-level introspection library a developer builds custom
  DPU-side tooling on top of. It is **not covered by this bundle**
  (policy-excluded from the public release); when Argus is
  genuinely insufficient and the team needs to build their own
  security product, route to the public docs via
  `doca-public-knowledge-map/SKILL.md`.
  The path-selection rule in
  the capabilities companion
  routes the user to Argus first for production security.
- `doca-dms/SKILL.md` and
  `doca-firefly/SKILL.md` — sibling service
  skills with the same service-skill shape (container on
  BlueField Arm, Container Deployment Guide recipe,
  smoke-before-bulk, env preconditions) over a different domain
  (DMS = device management via gNMI / gNOI; Firefly = time sync
  via PTP; Argus = runtime security via finding emission).
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Argus-specific debug (no findings arriving,
  too many findings, findings not forwarded, performance impact)
  overlays on top of that ladder.
