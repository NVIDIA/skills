---
license: Apache-2.0
name: doca-urom-svc
description: >
  SECURITY: the DOCA UROM Service binary has NO standalone access
  control (no authz list / `allowed_host` / `allowed_users` /
  `auth_token`) — access is governed ENTIRELY by DOCA Comch pairing +
  RDMA permissions, so any host able to establish a Comch pairing can
  drive remote memory ops; isolate accordingly. Use when operating the
  DOCA UROM Service container on a BlueField Arm to run remote memory
  ops (puts, gets, atomics, collectives) that paired hosts enqueue via
  the host-side `doca-urom` library — pulling the NGC container,
  choosing UCX-component surface, sizing the enqueue queue, wiring
  Comch endpoint pairing with `doca-urom`, or pinning host-library +
  service versions per the DOCA Compatibility Policy. Trigger even
  without 'DOCA UROM' — e.g. 'UCX collectives slow on host CPU, want
  BlueField offload', 'enqueue returns NOT_PERMITTED though doca_dev
  access is fine', 'ops enqueue but completions never fire'. Refuse
  for host-side app code, MPI/UCX integration, or installing DOCA.
metadata:
  kind: service
compatibility: >
  BlueField-Arm-only DOCA service container; pulled from NVIDIA
  NGC and started under the BlueField OS container runtime.
  Host-side install is irrelevant — the host's relationship to
  this service is via the paired `doca-urom` library over a
  `doca-rdma` substrate.
---

# DOCA UROM Service

**Where to start:** This skill is for *operating the DOCA UROM
Service container* on the BlueField Arm side. It is *not* for
*linking against* a library, and it is *not* the host-side
enqueue surface. If the user wants to *deploy* or *run* the
service container, open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure). If the question is *what
shape of service is DOCA UROM Service, what does it execute, and
how does it pair with the host-side library*, start at
[`CAPABILITIES.md`](CAPABILITIES.md). If DOCA is not installed
on the BlueField yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user's
real question is about *writing host-side code that enqueues
remote memory operations through the paired API*, the right
skill is [`doca-urom`](../../libs/doca-urom/SKILL.md) — the
host-side library; this service is the DPU-side executor that
library offloads to.

## Example questions this skill answers well

The CLASSES of DOCA UROM Service questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"Is DOCA UROM Service the right thing to deploy on my
  BlueField, or do I just need the host library?"** — worked
  example: *"my MPI cluster's host nodes link against
  `doca-urom`; what runs on the BlueField side and why must it
  also be there?"*. Answered by the publisher / executor
  paired-contract model in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the deploy-this-when path-selection rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Which library / service version pair am I supposed to
  run together?"** — worked example: *"the host fleet upgraded
  to a newer `doca-urom`; do I have to upgrade the service
  containers on every BlueField, or is the pairing flexible?"*.
  Answered by the version-contract overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the paired-version step in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`doca-version`](../../doca-version/SKILL.md) as the canonical
  body.
- **"What does the service configure — UCX components,
  collectives, queue depths, how the host pairs over Comch?"** — worked
  example: *"my upstream stack wants to offload all-reduce
  collectives; how do I tell the service to expose that
  collective family and how does the host pair to it over DOCA
  Comch?"*. Answered by the configuration-axes table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the config-authoring step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Host's `doca-urom` calls fail with `NOT_PERMITTED` even
  though `doca_dev` access is fine — is this the service?"** —
  worked example: *"first enqueue from host returns
  `DOCA_ERROR_NOT_PERMITTED` after a clean `doca_ctx_start()`"*.
  Answered by the Comch-pairing / RDMA-permissions layer in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug), which surfaces *"is
  the DOCA Comch endpoint pair correctly established and is the
  underlying RDMA permission stack happy"* BEFORE blaming a
  service-side authz layer (no such layer exists in the shipped
  binary — `NOT_PERMITTED` here is a Comch / RDMA signal, not a
  UROM-service authz signal).
- **"Operations enqueue but never complete — service or
  substrate?"** — worked example: *"host enqueue succeeds, the
  progress engine never sees the completion, what layer is
  hung"*. Answered by the service-vs-substrate split in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug), which separates
  *service queue full / handler stuck* from *underlying RDMA
  transport down* before recommending a fix on either side.
- **"Performance with offload is worse than the host-CPU
  baseline — is the service the bottleneck?"** — worked
  example: *"we deployed the service, the workload runs, but
  collectives are slower than when the host CPU posted them
  itself"*. Answered by the offload-isn't-free rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the smoke-before-scale step in
  [`TASKS.md ## test`](TASKS.md#test), which surfaces *the
  workload's pattern may not actually benefit from DPU offload*
  as a legitimate diagnosis, not a service bug.

## Audience

This skill serves **external operators and platform teams who
deploy and operate the DOCA UROM Service container** on
BlueField to receive and execute the remote memory operations
HPC / UCX / MPI workloads on the host enqueue through
`doca-urom`. Concretely: people running the service container
on BlueField Arm, choosing which UCX components and collectives
it exposes, sizing the enqueue queue depth, wiring the DOCA
Comch endpoint pairing between host `doca-urom` and the
service container (the shipped binary has NO standalone
service-side "host-endpoint authorization list" — access is
governed by Comch pairing + the underlying RDMA permissions),
and validating the host-library + DPU-service paired contract
end-to-end before scaling a real HPC workload on top.

It is **not** for NVIDIA developers contributing to the DOCA
UROM Service itself, and it is **not** a programming guide for
*building applications on top of* DOCA libraries (that is
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
plus the matching `libs/<library>` skill). DOCA UROM Service is
a **service**, not a library: the operator deploys a container
on the BlueField and configures it via the documented config
surface; they do not link `lib<uromservice>.so` to write their
own program. The paired host-side library
[`doca-urom`](../../libs/doca-urom/SKILL.md) is a separate
skill with its own scope and its own audience (HPC application
developers, not service operators); the agent must refuse to
collapse the library and the service into one another.

**Path selection up front.** Deploy this service when the HPC
cluster's host nodes use the `doca-urom` library and want host
CPU freed for compute by offloading collective communication
to the BlueField, when the team is building a custom HPC stack
on top of `doca-urom`, or when an upstream MPI / UCX stack has
been wired to use UROM as a transport. Do **not** deploy this
service when the hosts are not using `doca-urom`, when the HPC
stack is neither MPI nor UCX (this service won't help — it
executes UROM-shaped offloads, not arbitrary networking), or
when the BlueField hardware is too constrained for the
intended offload (a cap-query at deploy time surfaces this
upfront, not after the service is running). Deploying the
service speculatively into an environment whose host workloads
will not actually offload through `doca-urom` adds operational
complexity without any agent-visible benefit.

## When to load this skill

Load this skill when the user is doing **hands-on DOCA UROM
Service deployment work** on a BlueField where DOCA is already
installed. Concretely:

- Deciding *whether* DOCA UROM Service is the right answer for
  the user's HPC environment (vs. keeping the host CPU on the
  communication path with raw `doca-rdma` or with no DPU
  offload at all).
- Deploying the service container on BlueField Arm — pulling
  the image per the public DOCA UROM Service Guide, setting the
  daemon's CLI flags / env (`SERVICE_ARGS`, `UROM_PLUGIN_PATH`)
  and mounting the `plugins/` directory, starting / stopping
  the container under the BlueField container runtime per the
  public Container Deployment Guide.
- Choosing the service's configuration axes — which UCX
  components / collectives the service exposes (cap-bound to
  what the BlueField generation supports), enqueue queue
  depths for the offload path, and how the host's `doca-urom`
  library pairs to the service over DOCA Comch (access is
  governed by that Comch pairing + the underlying RDMA
  permissions — there is no service-side authorization list).
- Confirming the host-library + DPU-service version pair is
  one the DOCA Compatibility Policy supports — a mismatch is
  the canonical subtle-failure mode for the paired contract.
- Reading the service container's logs, the service's
  observability surface, and the underlying RDMA substrate
  counters to confirm the service is actually executing the
  operations the host enqueued.
- Debugging a deployment where the container is healthy but
  the host's `doca-urom` enqueues fail or never complete, or
  where the offload's performance is worse than the host-CPU
  baseline.

Do **not** load this skill for general DOCA orientation,
install of DOCA itself, host-side `doca-urom` library API
questions, or non-UROM HPC stack topics. For those, route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-setup`](../../doca-setup/SKILL.md), or the matching
host-side library skill
[`doca-urom`](../../libs/doca-urom/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — the service's architecture (long-running
  container on BlueField Arm that executes UROM offloads from
  paired hosts), the publisher / executor paired-contract
  model and its load-bearing version-coupling rule, the
  configuration axes (UCX-component / collective surface,
  enqueue queue sizing, DOCA Comch endpoint pairing), the deployment
  shape (container on BlueField Arm per the public Container
  Deployment Guide), the pairing surface (host `doca-urom`
  library + underlying `doca-rdma` transport substrate), the
  observability surface (container state + service logs + RDMA
  counters), the error taxonomy (container-runtime vs
  service-side-resource vs
  transport-substrate vs paired-version-mismatch), and the
  safety policy (path-selection rule, version-contract rule,
  smoke-before-scale).
- `TASKS.md` — step-by-step workflows for the in-scope service
  verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`, plus a `Deferred task verbs` block routing
  out-of-scope questions and a `Command appendix` of recurring
  commands.

The skill assumes a BlueField where DOCA is already installed
and the operator has the privileges the public DOCA UROM
Service Guide expects to pull, run, and configure containers
on BlueField Arm. It does not cover installing DOCA — that path
goes through [`doca-setup`](../../doca-setup/SKILL.md). It does
not cover the host-side `doca-urom` library API — that is
[`doca-urom`](../../libs/doca-urom/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or
sample-config bundle. To keep the boundary clean, it
deliberately does not contain — and pull requests should not
add:

- **Pre-baked DOCA UROM Service flag / env bundles**
  (full UCX-component / collective exposure manifests,
  ready-to-run queue-depth `SERVICE_ARGS` strings) intended
  to be copy-pasted into
  production. Service configuration is deployment-specific
  (per the BlueField
  generation's capability cap, per the workload's collective
  pattern); the safe answer for an external operator is to
  derive the daemon's flags / env from the public DOCA UROM
  Service Guide against their own deployment. The agent's job is to
  prescribe the *procedure* and the *configuration-axes
  decision*, not to ship a config the user might run
  unmodified.
- **Container image names, tags, or registry paths.** The
  authoritative image source is the public DOCA UROM Service
  Guide reachable through
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services);
  the service's image tag is version-bound and changes
  between DOCA releases AND must match the host-side library
  version per the paired-contract rule. Inventing or
  memorizing a tag is the canonical hallucination failure
  mode for a service skill, and for this service in
  particular it can silently produce a host-library / service
  version mismatch.
- **Host-side `doca-urom` application source code, build
  manifests, or MPI / UCX integration glue.** Those live on
  the host and belong to
  [`doca-urom`](../../libs/doca-urom/SKILL.md) (host library
  API) or to upstream MPI / UCX documentation (stack-side
  integration). This skill names *that* the host side must
  be wired through `doca-urom` and *that* the version must
  pair with the service; the host-side bodies are out of
  scope.
- **A `samples/`, `templates/`, or `reference/` subtree** of
  any kind. A mock or incomplete artifact in this skill's
  tree, even one labeled *"reference"*, is misleading:
  operators will read it as production-ready, and for a
  paired-contract service that risk is amplified because
  *"production-ready"* implicitly claims a host-library
  version pairing that the skill cannot guarantee for the
  reader.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope **and** that the DOCA UROM Service is the
   right answer at all (vs. routing entirely to the host-side
   library skill, or away from UROM if the workload doesn't
   actually benefit from offload).
2. **For the service's deployment shape, the publisher /
   executor paired-contract model, the configuration axes,
   the host-library + RDMA-substrate pairing surface, the
   error taxonomy, the observability surface, the
   version-coupling rule, and the path-selection safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-urom`](../../libs/doca-urom/SKILL.md) — the paired
  host-side library. Hosts link `doca-urom` to ENQUEUE remote
  memory operations; this service EXECUTES them. The two
  skills load together for any HPC offload deployment and they
  do NOT collapse into one another: the library never executes
  on its own; this service is never enqueued through by
  itself. Mismatches between the library and the service are
  the dominant subtle-failure mode for the paired contract.
- [`doca-rdma`](../../libs/doca-rdma/SKILL.md) — the underlying
  RDMA transport substrate this service uses to actually move
  bytes once a host enqueue lands on the DPU. The service does
  NOT replace RDMA; it sits on top of it. A failing RDMA fabric
  surfaces at the service as *operations enqueued but never
  complete* and at the host as `DOCA_ERROR_IO_FAILED`; the fix
  is on the substrate side, not in this service's config.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public DOCA UROM Service Guide
  and the rest of the public DOCA documentation set. The
  service URL is listed under
  [`## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation
  and install verification on the BlueField where the service
  container will run, including the *I have no install yet*
  path via the public NGC DOCA container. This skill assumes
  its preconditions are satisfied on BlueField Arm.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This service's container tag is
  version-bound AND its host-library pairing is version-bound;
  this skill's `## Version compatibility` cross-links the
  four-way match rule and adds the host-library + DPU-service
  paired-version overlay that is load-bearing for UROM.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA patterns. The DOCA UROM Service is
  service-shaped not library-shaped, so the build / modify /
  first-app pattern there does not apply directly, but the
  cross-library debug discipline (env-before-program,
  layer-before-config) remains useful when the service
  reports an error that originated in the container runtime
  or in a DOCA library it called.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Service-specific debug (container not
  running, host-library / service version mismatch,
  transport substrate down, offload
  not actually helping) overlays on top of that ladder.
- [`doca-dms`](../doca-dms/SKILL.md) and
  [`doca-firefly`](../doca-firefly/SKILL.md) — sibling service
  skills. The agent reading these skills should see the same
  service-skill shape (container on BlueField Arm, public
  Container Deployment Guide as the canonical recipe, env
  preconditions checked first, configured via the documented
  per-service surface — for UROM that is the daemon's CLI flags
  / env, not a mounted config file — smoke-before-scale)
  layered on top of a different
  per-service problem domain (DMS = device management via
  gNMI / gNOI; Firefly = time synchronization via PTP;
  UROM Service = HPC remote memory operation execution via
  the paired `doca-urom` library).
