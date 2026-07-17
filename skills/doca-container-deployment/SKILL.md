---
license: Apache-2.0
name: doca-container-deployment
description: >
  Use this skill when the user is hands-on deploying an in-bundle DOCA
  service container (Argus, DMS, Firefly, Flow-Inspector, OS-Inspector,
  UROM service) on a BlueField — kubelet standalone watching a
  static-pod manifests directory, YAML pod-spec drop, kubelet status /
  ENTRYPOINT logs / per-service liveness, smoke-before-bulk, and the
  layered error taxonomy (pod-spec, scheduling, image pull, runtime,
  mount, network, version, host). Trigger even when the user does not
  say "container deployment" — typical implicit phrasings include "how
  do I run my built service on the BlueField?", "where do I drop the
  pod-spec YAML?", "pod stuck in Pending / ImagePullBackOff /
  CrashLoopBackOff", "container Running but service isn't ready", "pod
  restart-loops after edit", or "DMS and Firefly together". Refuse and
  route elsewhere for per-service config schemas, DOCA install,
  library-API questions, external NVIDIA services (BlueMan, HBN, SNAP,
  Virtio-net), or full Kubernetes-cluster ops — those belong to other
  skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is an overlay loaded
  against any DOCA artifact skill); the validation steps within DO
  require a live DOCA install at /opt/mellanox/doca.
---

# DOCA container deployment

**Where to start:** This skill is for *operating* the cross-cutting
DOCA container-deployment runtime — the shared pattern every DOCA
service on the BlueField uses to come up (kubelet standalone agent
on the BlueField Arm watching a static-pod manifests directory; the
operator drops a YAML pod spec into that directory; kubelet schedules
the pod and runs the container).

**If the developer has NOT yet decided container vs. bare-metal**
(*"I just got a BlueField, what now?"*, *"my code is built, how do I
run it?"*, *"how do I deploy this?"*), route them BACK to
[`doca-setup ## recognize`](../doca-setup/TASKS.md#recognize) first.
That is the front-door routing decision. The wrong failure mode is
to silently push every developer onto the container path because the
agent loaded this skill first. `## recognize` detects the system
shape, asks the minimum residual question, and lands the developer on
either this skill (when the workload is a packaged DOCA service to
drop on a BlueField) or the bare-metal-path sibling
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)
(when the workload is a DOCA-linked application binary the developer
launches directly).

**If the developer is already on the container path**, open
[`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure). If the question is *what shape
of runtime is this and what does the deployment contract look like*,
start at [`CAPABILITIES.md`](CAPABILITIES.md). For per-service
overlays, follow the per-service skill under `skills/services/` that
layers on top of this one — the DOCA monorepo ships six service
skills (Argus, DMS, Firefly, Flow-Inspector, OS-Inspector, UROM
service); externally-productized NVIDIA services (BlueMan, HBN, SNAP,
Virtio-net, DOCA Telemetry Service as productized, …) are out of
scope for this bundle by the strict-to-DOCA invariant — route those
to the public NVIDIA docs at `docs.nvidia.com/doca/sdk/`. If DOCA is
not installed on the BlueField target yet, route to
[`doca-setup`](../doca-setup/SKILL.md) first.

## Audience

This skill serves **external operators and platform teams who deploy
DOCA service containers on BlueField** — i.e., people who have a
BlueField with DOCA installed on the Arm side, a container runtime
plus the kubelet standalone agent already present per the BlueField
OS image, and the host-OS permissions the public DOCA Container
Deployment Guide names for the chosen service. The skill is the
shared deployment runtime; each per-service skill in the bundle
(see the list in [`## Related skills`](#related-skills)) supplies
the service-specific config schema, paired-workload contract, and
"healthy" definition.

It is **not** for NVIDIA developers contributing to the BlueField
container runtime or to kubelet itself, and it is **not** a generic
Kubernetes tutorial. Kubelet runs on the BlueField in *standalone*
mode here — no full Kubernetes control plane, no `kubectl` against
a cluster API server — and the substantive answer to most
container-deployment questions on the BlueField is the public DOCA
Container Deployment Guide. This skill teaches the agent which
guide to quote, in what order to walk it, and how to map a symptom
to a layer; it does NOT re-invent kubelet flags, pod-spec field
names, or static-pod path strings. The shared deployment runtime
described here is the cross-cutting layer; the per-service skill
(`doca-argus`, `doca-dms`, `doca-firefly`,
`doca-urom-svc`) supplies the per-service
config schema, paired-workload contract, and "healthy" definition.

## When to load this skill

Load this skill when the user is doing **hands-on container
deployment of any DOCA service** on a BlueField target, or asking a
cross-service deployment question that is not specific to one
service's config schema. Concretely:

- Dropping a YAML pod spec into the documented static-pod manifests
  directory on the BlueField Arm so kubelet standalone schedules the
  pod and runs the DOCA service container.
- Inspecting pod status, container logs, and the documented
  liveness signal for any in-bundle DOCA service container — Argus,
  DMS, Firefly, Flow-Inspector, OS-Inspector, UROM service — so the
  agent answers "did the container come up, and is the service
  inside actually ready" the same way for every service.
- Walking the smoke-before-bulk loop (pod reaches `Running`;
  ENTRYPOINT logs are clean; service answers a trivial liveness
  probe) BEFORE the BlueField is put under workload.
- Diagnosing a deployment that is misbehaving — pod-spec YAML is in
  the directory but the pod never schedules; pod schedules but
  image-pull fails; image pulls but container ENTRYPOINT
  immediately exits; container runs but the service inside never
  answers; container is in a restart loop after a config edit; a
  volume mount the pod spec names is missing on the host; a network
  policy or host-firewall rule is blocking the service.
- Cross-service questions: *"can I have DMS and Firefly on the
  same BlueField"*, *"how do I list every DOCA service pod that is
  currently running"*, *"what is the documented stop / restart
  semantics if I edit a pod-spec file in place"*.

Do **not** load this skill for per-service config schema questions
(those belong to the matching per-service skill); for installing
DOCA itself or preparing the BlueField env (use
[`doca-setup`](../doca-setup/SKILL.md)); for library-API
questions (use the matching `libs/<library>` skill); or for general
Kubernetes-cluster operations (this skill covers kubelet *standalone*
mode on the BlueField, not a full Kubernetes control plane).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — the cross-cutting DOCA container-deployment
  runtime contract on the BlueField (kubelet standalone agent on
  BlueField Arm watching a documented static-pod manifests
  directory; YAML pod-spec drop is the unit of operator input; the
  same pattern applies across every DOCA service), the BlueField
  preconditions (DOCA install, container runtime, BFB version,
  per-service firmware slot when the service emulates a device,
  image-pull reachability to NGC, host-OS permissions), the
  observability surface (kubelet status, container logs, service-
  side liveness signal — three layers, each with its own owner),
  the cross-cutting error taxonomy (pod-spec syntax → pod
  scheduling → image pull → runtime → volume mount → network policy
  → version → cross-cutting host) covering ≥ 6 layers, and the
  safety policy (smoke before bulk; failed pod is high-stakes —
  clear the root cause before letting kubelet restart-loop the
  pod; do NOT invent pod-spec field names / kubelet flags / image
  tags from memory).
- `TASKS.md` — step-by-step workflows for the in-scope deployment
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`,
  plus a `Deferred task verbs` block routing per-service config
  questions, host-firmware-slot work, paired-workload work, and
  full-Kubernetes-cluster work out to their owning skills.

The skill assumes a BlueField target where DOCA is already installed
on the Arm side, the BlueField OS image ships kubelet standalone +
the container runtime per the public DOCA Container Deployment
Guide, and the operator has the host-OS permissions that guide
names. It does not cover installing DOCA — that path goes through
[`doca-setup`](../doca-setup/SKILL.md) — and it does not
re-document the per-service config schema, which is the canonical
concern of each DOCA service's public guide reached through
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (cross-cutting deployment runtime, NOT a per-service
   config-schema question).
2. **For the kubelet-standalone-mode runtime shape, the static-pod
   manifests directory rule, the host-OS / BFB / firmware-slot /
   image-pull preconditions, the error taxonomy (≥ 6 layers), the
   observability surface, and the safety / smoke-before-bulk
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

## Example questions this skill answers well

See [`references/details.md`](references/details.md#example-questions-this-skill-answers-well).
## What this skill deliberately does not ship

See [`references/details.md`](references/details.md#what-this-skill-deliberately-does-not-ship).
## Related skills

See [`references/details.md`](references/details.md#related-skills).
