---
license: Apache-2.0
name: doca-urom
description: >
  Use this skill when the user is doing hands-on DOCA UROM library work
  from the host side — wiring doca-urom under an HPC / UCX / MPI stack
  to OFFLOAD remote memory operations (puts, gets, atomics, collectives)
  onto a BlueField DPU, creating a UROM Service context
  (doca_urom_service_*) and Worker contexts (doca_urom_worker_*) that
  run plugins on the DPU, discovering plugins via
  doca_urom_service_get_plugins_list, progressing completions, or
  debugging DOCA_ERROR_* from a doca_urom_* call. Trigger even without
  "DOCA UROM": "MPI all-reduce burning host CPU", "push UCX traffic
  onto the BlueField", "first doca_urom call returns NOT_PERMITTED", or
  "host library and DPU service look out of sync". Route elsewhere for
  UROM Service deployment on the DPU side, MPI / UCX collective
  algorithm design, and RDMA / RoCE / IB substrate bring-up.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) on a host paired with a BlueField DPU
  running the DOCA UROM Service at a compatible version. Reads the
  user's local install via `pkg-config doca-urom` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}; underlying
  RDMA fabric between host and BlueField must be healthy.
---

# DOCA UROM

**Where to start:** This skill assumes DOCA is already installed on
both the host and the BlueField, the DOCA UROM Service is
deployed and running on the BlueField side, and the user is doing
**hands-on UROM work from the host side** — i.e. using
`doca-urom` from an HPC / UCX / MPI stack on the host to enqueue
remote memory operations (puts, gets, atomics, active messages,
collective primitives) that the BlueField DPU will execute on the
host's behalf. Open [`TASKS.md`](TASKS.md) if the user wants to
*do* something (configure / build / modify / run / test /
debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what can the host-side UROM API express* on this
version + this BlueField + this UROM Service version. If the
user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user is
asking about the **DPU-side UROM Service** itself (deployment,
container, operation lifecycle on the DPU side), that is a
DIFFERENT artifact — route via
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
to the public *DOCA UROM Service* guide. This skill is the
**host-side library**; the UROM Service is the **DPU-side
executor**, and they are a paired contract.

## Example questions this skill answers well

The CLASSES of UROM questions this skill is built to answer,
each with one worked example. The agent should treat the *class*
as the load-bearing piece — the worked example is a single
instance.

- **"How do I offload my MPI / UCX remote memory operations from
  the host CPU to the BlueField DPU?"** — worked example: *"my
  MPI all-reduce is consuming host CPU cycles I'd rather use for
  compute — how do I push that work onto the BlueField via
  UROM?"*. Answered by the host-library-plus-DPU-service
  paired-contract model in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the host-side bring-up workflow in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Is the DOCA UROM Service even running on my BlueField, and
  why does that matter before I write any `doca_urom_*` code?"** —
  worked example: *"my first `doca_urom_*` call returns
  `DOCA_ERROR_NOT_PERMITTED` on a host where DOCA is otherwise
  healthy"*. Answered by the env-precondition matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the service-deployed-and-running check in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes service-side env questions to
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
- **"Is this UROM operation type / atomic / collective supported
  on my device + this DOCA install + this UROM Service
  version?"** — worked example: *"does my BlueField support
  remote atomic Fetch-and-Add for an MPI window?"*. Answered by
  the plugin-discovery rule (`doca_urom_service_get_plugins_list`
  on a started Service — UROM operations are plugin-defined
  Command tasks, so the supported-plugins list is the
  capability surface) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How does `doca-urom` relate to `doca-rdma` — am I replacing
  it, layering on top, or something else?"** — worked example:
  *"I already have raw `doca-rdma` working; should I rewrite to
  use UROM, or is that the wrong tool?"*. Answered by the
  path-selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (UROM uses the RDMA transport substrate underneath but adds
  the DPU-offload contract on top, and is the right tool only
  when host CPU is the bottleneck due to communication
  overhead — small / simple point-to-point cases stay on
  `doca-rdma`).
- **"Is this UROM API on my installed DOCA version?"** — worked
  example: *"is the collective-ops plugin discoverable via
  `doca_urom_service_get_plugins_list` on DOCA 3.x"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  UROM-specific *host library and DPU service versions must
  match* overlay.
- **"What does this `DOCA_ERROR_*` from a `doca_urom_*` call
  mean and which layer caused it?"** — worked example:
  *"`DOCA_ERROR_NOT_PERMITTED` on the first
  `doca_urom_*` enqueue after `doca_ctx_start()` succeeded"*.
  Answered by the UROM overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building HPC / UCX / MPI
applications that consume the DOCA UROM library from the host
side** — i.e., users whose code calls `doca_urom_*` (directly in
C / C++, or through FFI / bindings from another language, or
through a UCX-based stack such as OpenMPI / MPICH that has been
wired to use DOCA UROM as a UCX transport) to push remote memory
operations onto the BlueField DPU instead of executing them on
the host CPU. It is *not* for NVIDIA developers contributing to
DOCA UROM itself, nor is it the place to learn how to **deploy /
operate the DOCA UROM Service** on the DPU side — that goes
through the public *DOCA UROM Service* guide via
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

**Language scope.** DOCA UROM ships as a host-side C library
with `pkg-config` module name `doca-urom`. The shipped samples
under `/opt/mellanox/doca/samples/doca_urom/` are written in C
(NVIDIA's choice). C and C++ consumers — including UCX-based
stacks that wrap the library — are the canonical case and the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the lifecycle, capability-discovery,
service-deployed-and-running, error-taxonomy, and
RDMA-substrate guidance language-neutral, and to route the agent
to the public C ABI as the authoritative surface that any
wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA UROM work
**from the host side**, in any language. Concretely:

- Initializing a UROM Service context (`doca_urom_service_*`) on
  a `doca_dev` that maps to the BlueField the user wants to
  offload to, then creating Worker contexts
  (`doca_urom_worker_*`) attached to that Service, and confirming
  the matching DPU-side UROM Service is reachable before the
  first enqueue.
- Enqueueing remote memory operations (puts, gets, atomics,
  active messages, collective primitives) through the host-side
  `doca_urom_*` API and progressing the DOCA progress engine
  for completions.
- Reading or setting library properties via the host-side UROM
  API and calling `doca_urom_service_get_plugins_list` on a
  started Service to discover which plugins (and therefore which
  operation types / atomics / collectives, since these are
  plugin-defined) this device + this DOCA install + this DPU-side
  UROM Service version actually supports.
- Wiring the host-side `doca-urom` library underneath a
  UCX-based HPC stack (OpenMPI, MPICH, custom UCX consumer) so
  the stack's remote memory traffic offloads to the BlueField
  DPU instead of running on the host CPU.
- Debugging a `DOCA_ERROR_*` returned from a `doca_urom_*` call
  — in particular disambiguating *DPU-side UROM Service not
  reachable* from *operation type not supported on this device*
  from *standard `doca_dev` access denied* from *underlying
  RDMA transport failure*.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the UROM C ABI — for the lifecycle,
  service-deployed-and-running, capability-discovery, and
  error-taxonomy rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, deployment / operation of the DOCA UROM Service
on the DPU side (a separate artifact, with its own public guide
reachable via
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)),
or non-UROM library questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
UROM-specific material lives in two companion files:

- `CAPABILITIES.md` — what the host-side UROM API can express on
  this version + this BlueField + this UROM Service version: the
  paired-contract model (host library enqueues; DPU service
  executes); the two host-side context types — the
  `doca_urom_service` (one per BlueField, bound to its
  `doca_dev`) and the `doca_urom_worker` contexts attached to it;
  the enqueue-side operation surface (puts, gets, atomics, active
  messages, collective primitives) delivered as plugin-defined
  Worker Command tasks and named generically because exact
  symbol shapes are plugin- and install-bound; the
  plugin-discovery surface (`doca_urom_service_get_plugins_list`);
  the UROM error taxonomy mapped onto the cross-library `DOCA_ERROR_*` set;
  the observability surface (completion events on the DOCA
  progress engine, capability snapshots, infrastructure-side
  RDMA counters); and the safety policy that gates env
  preconditions (DOCA UROM Service deployed and running on the
  DPU side; host library and DPU service versions agreeing; an
  RDMA-capable BlueField + DOCA install).
- `TASKS.md` — step-by-step workflows for the six in-scope UROM
  verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host + BlueField pair where DOCA is already
installed at the standard location, the DOCA UROM Service is
already deployed and running on the BlueField, the underlying
RDMA transport between host and BlueField is healthy, and the
user already has at least a sketch of the HPC / UCX / MPI stack
they want to offload. It does not cover installing DOCA,
deploying the UROM Service container on the BlueField, or
bringing up RDMA / RoCE / IB transport between host and
BlueField — those paths go through
[`doca-setup`](../../doca-setup/SKILL.md), the public *DOCA
UROM Service* guide, and [`doca-rdma`](../doca-rdma/SKILL.md)
respectively.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA UROM application source code, in any
  language.** The verified UROM source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_urom/`. The
  agent's job is to route the user to those files and prescribe
  a minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the UROM-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **An MPI / UCX algorithm or collective implementation.** This
  library *offloads* remote memory operations; the algorithm
  selecting which puts / gets / atomics / collectives to issue
  for a given MPI primitive is the user's HPC stack (OpenMPI,
  MPICH, custom UCX consumer) or the user's domain expertise.
  The agent must refuse to invent collective algorithms and
  must route any *"what algorithm should my all-reduce use"*
  question to the upstream MPI / UCX documentation — that is a
  research / stack-design question, not a UROM API question.
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-urom` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will read
  it as buildable.
- **The DOCA UROM Service surface.** That service is a
  *separate artifact* with its own public guide; routing for
  its deployment / operation lives in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
  Conflating the library (this skill, host-side enqueue) with
  the service (DPU-side executor) is the single most common
  UROM first-app design error.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (host-side library work; not DPU-side service
   deployment, and not MPI / UCX algorithm design).
2. **For the UROM capability matrix, the paired-contract model
   (host library + DPU service), the Service + Worker context
   model, the enqueue-side operation surface, the
   plugin-discovery rule, the env-precondition policy, the
   error taxonomy, the observability surface, and the safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules (with the UROM overlay that the
host library version and the DPU service version must match),
[`doca-rdma`](../doca-rdma/SKILL.md) for the underlying RDMA
transport substrate UROM offloads ride on top of, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
UROM library guide, the public DOCA UROM Service guide, or in
the on-disk install layout" rather than "UROM
host-side-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  DOCA UROM library public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-UROM/index.html>; the
  DOCA UROM Service is a *different artifact* documented
  separately under the *DOCA services* section of that map.
- [`doca-rdma`](../doca-rdma/SKILL.md) — the underlying RDMA
  transport substrate the BlueField uses to actually move
  bytes between nodes once UROM has offloaded a remote memory
  operation. UROM does NOT replace RDMA; it adds the
  DPU-offload contract on top so the host CPU does not have to
  post the verbs itself. When the user's intent is simple
  point-to-point RDMA and the host CPU is not the bottleneck,
  the right tool is `doca-rdma` directly — UROM adds the
  service-side contract that isn't worth its overhead for that
  case.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, BlueField setup, and the *I have no
  install yet* path with the public NGC DOCA container. This
  skill assumes its preconditions are satisfied AND that the
  DPU-side UROM Service is deployed and running on the
  BlueField.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the UROM-specific *host library and DPU service must agree on
  version* overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal Core-context lifecycle, the cross-library
  `DOCA_ERROR_*` taxonomy, and the program-side debug order.
  This skill layers UROM specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). UROM-specific debug (DPU-side UROM
  Service not running / not reachable, host library + DPU
  service version skew, RDMA transport failure underneath the
  UROM offload, operation type not in this device + firmware +
  service version) overlays on top of that ladder.

The **DOCA UROM Service** that runs on the DPU side is
deliberately **not in scope** for this skill — it is a separate
artifact with its own public guide, reachable through
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
Conflating the library and the service is the single most
common UROM first-app design error: the agent must surface the
library / service split explicitly whenever the question
straddles the two.
