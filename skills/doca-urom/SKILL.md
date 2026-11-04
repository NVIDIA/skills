---
license: Apache-2.0
name: doca-urom
description: >
  Host-side DOCA UROM library (doca_urom_*): offload an HPC / UCX / MPI
  application's remote memory operations (puts, gets, atomics,
  collectives) from the host CPU onto a BlueField DPU. Trigger on "MPI
  all-reduce burning host CPU", "push UCX traffic onto the BlueField",
  "first doca_urom call returns NOT_PERMITTED", or "host library and DPU
  service look out of sync", even without naming DOCA UROM. An offload
  needs both halves: this host library and the UROM Service on the DPU
  (doca-urom-svc), which owns deploying that container. Covers creating
  UROM Service contexts (doca_urom_service_*) and Worker contexts
  (doca_urom_worker_*) that run plugins on the DPU, discovering plugins
  via doca_urom_service_get_plugins_list, progressing completions, and
  DOCA_ERROR_* from doca_urom_* calls. Route elsewhere for UROM Service
  deployment on the DPU side, MPI / UCX collective algorithm design, and
  RDMA / RoCE / IB substrate bring-up.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - urom
    - hpc
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) on a host paired with a BlueField DPU
  running the DOCA UROM Service at a compatible version. Reads the
  user's local install via `pkg-config doca-urom` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}; underlying RDMA
  fabric between host and BlueField must be healthy. The validation
  steps require a live DOCA install.
---

# DOCA UROM

## Examples

- **"Push UCX traffic onto the BlueField."** — this is in
  scope. Stay on host-side `doca-urom`; do not deploy the
  DPU-side UROM Service here.
- **"First doca_urom call returns NOT_PERMITTED."** — check
  host/service version alignment and the RDMA substrate first.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
offload / debug. Open the capabilities companion for service vs
worker contexts. Do not open every companion file.

**Where to start:** This skill assumes DOCA is installed on both
host and BlueField, the DOCA UROM Service is deployed and running
on the BlueField, and the user is doing **hands-on UROM work from
the host side** — using `doca-urom` from an HPC / UCX / MPI stack
to enqueue remote memory operations (puts, gets, atomics, active
messages, collectives) the DPU executes on the host's behalf. Open
the tasks companion to *do* something (configure / build / modify /
run / test / debug); open the capabilities companion for *what the
host-side UROM API expresses* on this version + BlueField + UROM
Service version. If DOCA is not installed, route to
`doca-setup/SKILL.md` first; for the **DPU-side
UROM Service** itself (deployment, container, lifecycle), that is a
DIFFERENT artifact — route via
`doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`)
to the public *DOCA UROM Service* guide. This skill is the
**host-side library**; the UROM Service is the **DPU-side
executor**; they are a paired contract.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. For
out-of-scope requests, name the skill that owns them and stop.

## Example questions this skill answers well

The CLASSES of UROM questions this skill answers, each with one
worked example. The *class* is the load-bearing piece; the worked
example is one instance.

- **"How do I offload my MPI / UCX remote memory operations from
  the host CPU to the BlueField?"** — worked example: *"my MPI
  all-reduce burns host CPU cycles I'd rather use for compute —
  how do I push that onto the BlueField via UROM?"*. Answered by
  the paired-contract model in
  the capabilities companion
  + the host-side bring-up workflow in
  the tasks companion.
- **"Is the DOCA UROM Service even running on my BlueField, and
  why does that matter before I write `doca_urom_*` code?"** —
  worked example: *"my first `doca_urom_*` call returns
  `DOCA_ERROR_NOT_PERMITTED` on an otherwise-healthy host"*.
  Answered by the env-precondition matrix in
  the capabilities companion
  + the service-deployed check in
  the tasks companion step 1, which
  routes service-side env questions to
  `doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`).
- **"Is this UROM operation type / atomic / collective supported
  on my device + DOCA install + UROM Service version?"** — worked
  example: *"does my BlueField support remote atomic Fetch-and-Add
  for an MPI window?"*. Answered by the plugin-discovery rule
  (`doca_urom_service_get_plugins_list` on a started Service —
  UROM operations are plugin-defined Command tasks, so the
  plugins list is the capability surface) in
  the capabilities companion
  + the discovery step in
  the tasks companion.
- **"How does `doca-urom` relate to `doca-rdma` — am I replacing
  it, layering on top, or something else?"** — worked example:
  *"I already have raw `doca-rdma` working; should I rewrite to
  use UROM?"*. Answered by the path-selection rule in
  the capabilities companion
  (UROM rides the RDMA substrate but adds the DPU-offload
  contract; it is the right tool only when host CPU is the
  bottleneck — simple point-to-point cases stay on `doca-rdma`).
- **"Is this UROM API on my installed DOCA version?"** — worked
  example: *"is the collective-ops plugin discoverable via
  `doca_urom_service_get_plugins_list` on DOCA 3.x"*. Answered by
  the version-compatibility overlay in
  the capabilities companion,
  which cross-links the canonical detection chain in
  `doca-version/SKILL.md` and adds the
  *host library + DPU service versions must match* overlay.
- **"What does this `DOCA_ERROR_*` from a `doca_urom_*` call
  mean and which layer caused it?"** — worked example:
  *"`DOCA_ERROR_NOT_PERMITTED` on the first enqueue after
  `doca_ctx_start()` succeeded"*. Answered by the UROM overlay on
  the cross-library taxonomy in
  the capabilities companion
  + the layered ladder in
  the tasks companion that escalates to
  `doca-debug/SKILL.md`.

## Audience

This skill serves **external developers building HPC / UCX / MPI
applications that consume the DOCA UROM library from the host
side** — code calling `doca_urom_*` (directly in C / C++, through
FFI / bindings, or through a UCX-based stack such as OpenMPI /
MPICH wired to use UROM as a transport) to push remote memory
operations onto the BlueField instead of the host CPU. It is *not*
for NVIDIA developers contributing to DOCA UROM, nor the place to
learn how to **deploy / operate the UROM Service** on the DPU side
— that goes through the public *DOCA UROM Service* guide via
`doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`).

**Language scope.** DOCA UROM ships as a host-side C library
(`pkg-config` module `doca-urom`); shipped samples under
`/opt/mellanox/doca/samples/doca_urom/` are C. C / C++ consumers —
including UCX-based stacks wrapping the library — are the
canonical case the tasks companion assumes. Other-language
consumers (Rust, Go, Python, …) reach the same `*.so` through FFI;
the skill keeps its lifecycle, capability-discovery,
error-taxonomy, and RDMA-substrate guidance language-neutral and
routes to the public C ABI as the authoritative surface.

## When to load this skill

Load this skill when the user is doing hands-on DOCA UROM work
**from the host side**, in any language. Concretely:

- Initializing a UROM Service context (`doca_urom_service_*`) on
  a `doca_dev` mapping to the target BlueField, creating Worker
  contexts (`doca_urom_worker_*`) attached to it, and confirming
  the matching DPU-side UROM Service is reachable before the
  first enqueue.
- Enqueueing remote memory operations (puts, gets, atomics,
  active messages, collectives) through `doca_urom_*` and
  progressing the DOCA progress engine for completions.
- Calling `doca_urom_service_get_plugins_list` on a started
  Service to discover which plugins — and therefore which
  operation types / atomics / collectives, since these are
  plugin-defined — this device + DOCA install + UROM Service
  version supports.
- Applying the UROM lifecycle, capability-discovery, and error
  rules to a UCX-based HPC stack (OpenMPI, MPICH, custom UCX
  consumer) already wired to use UROM. Designing the UCX
  transport integration itself remains upstream-stack work.
- Debugging a `DOCA_ERROR_*` from `doca_urom_*` — disambiguating
  *DPU-side UROM Service unreachable* vs *operation type
  unsupported on this device* vs *plain `doca_dev` access denied*
  vs *underlying RDMA transport failure*.
- Designing non-C bindings (Rust, Go, Python, …) that wrap the
  UROM C ABI and must honor the same lifecycle and error rules.

Do **not** load this skill for general DOCA orientation, DOCA
install, deployment / operation of the DPU-side UROM Service (a
separate artifact with its own public guide via
`doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`)),
or non-UROM library questions. For those, use
`doca-public-knowledge-map/SKILL.md`.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
UROM-specific material lives in two companion files:

- the capabilities companion — what the host-side UROM API expresses on
  this version + BlueField + UROM Service version: the
  paired-contract model (host enqueues; DPU service executes); the
  two context types (`doca_urom_service` — one per BlueField, bound
  to its `doca_dev` — and attached `doca_urom_worker` contexts);
  the enqueue-side operation surface (puts, gets, atomics, active
  messages, collectives) delivered as plugin-defined Worker Command
  tasks, named generically because exact symbol shapes are
  plugin- and install-bound; the plugin-discovery surface
  (`doca_urom_service_get_plugins_list`); the UROM error taxonomy
  mapped onto `DOCA_ERROR_*`; the observability surface
  (completion events, capability snapshots, RDMA counters); and the
  safety policy gating env preconditions (UROM Service deployed on
  the DPU; host library + DPU service versions agree; RDMA-capable
  BlueField + DOCA install).
- the tasks companion — step-by-step workflows for the six in-scope UROM
  verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host + BlueField pair where DOCA is installed,
the UROM Service is deployed and running on the BlueField, the RDMA
transport between them is healthy, and the user has a sketch of the
HPC / UCX / MPI stack to offload. It does not cover installing
DOCA, deploying the UROM Service container, or bringing up
RDMA / RoCE / IB transport — those go through
`doca-setup/SKILL.md`, the public *DOCA UROM
Service* guide, and `doca-rdma/SKILL.md`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA UROM application source code, in any
  language.** The verified UROM source is the shipped C samples at
  `/opt/mellanox/doca/samples/doca_urom/`. The agent routes the
  user there and prescribes a minimum-diff modification via the
  modify-a-sample workflow in
  `doca-programming-guide/SKILL.md`,
  layered with the UROM overrides in the tasks companion.
- **An MPI / UCX algorithm or collective implementation.** This
  library *offloads* remote memory operations; picking which
  puts / gets / atomics / collectives to issue for a given MPI
  primitive is the user's HPC stack or domain expertise. Refuse
  to invent collective algorithms; route *"what algorithm should
  my all-reduce use"* to upstream MPI / UCX docs — that is a
  stack-design question, not a UROM API question.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the manifest *in the user's project* against the installed DOCA,
  where `pkg-config --modversion doca-urom` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock artifact labeled "reference" reads as buildable and
  misleads.
- **The DOCA UROM Service surface.** That service is a *separate
  artifact* with its own public guide; routing lives in
  `doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`).
  Conflating the library (host-side enqueue) with the service
  (DPU-side executor) is the most common UROM first-app design
  error.

## Loading order

1. Read this skill first to confirm the user's question is
   in scope (host-side library work; not DPU-side service
   deployment, and not MPI / UCX algorithm design).
2. **For the UROM capability matrix, the paired-contract model
   (host library + DPU service), the Service + Worker context
   model, the enqueue-side operation surface, the
   plugin-discovery rule, the env-precondition policy, the
   error taxonomy, the observability surface, and the safety
   policy, see the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see the tasks companion.**

Both companion files cross-link to each other, to
`doca-version/SKILL.md` for canonical
version rules (with the UROM overlay that host library and DPU
service versions must match), to
`doca-rdma/SKILL.md` for the underlying RDMA
substrate, and to
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public UROM
library / Service guide or the on-disk install layout".

## Related skills

- `doca-public-knowledge-map/SKILL.md` —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  DOCA UROM library public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-UROM/index.html>; the
  DOCA UROM Service is a *different artifact* documented
  separately under the *DOCA services* section of that map.
- `doca-rdma/SKILL.md` — the underlying RDMA
  transport substrate the BlueField uses to move bytes once UROM
  offloads an operation. UROM does NOT replace RDMA; it adds the
  DPU-offload contract on top. For simple point-to-point RDMA
  where host CPU is not the bottleneck, use `doca-rdma` directly —
  UROM's service-side contract is not worth its overhead there.
- `doca-setup/SKILL.md` — env preparation,
  install verification, BlueField setup, and the *no install yet*
  path via the public NGC DOCA container. This skill assumes its
  preconditions hold AND the DPU-side UROM Service is deployed and
  running.
- `doca-version/SKILL.md` — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds the
  *host library + DPU service must agree on version* overlay.
- `doca-structured-tools-contract/SKILL.md` —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  the tasks companion honors this contract.
- `doca-programming-guide/SKILL.md` —
  general DOCA programming patterns shared by every library:
  `pkg-config` + meson build, modify-a-shipped-sample first-app,
  Core-context lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the debug order. This skill layers UROM specifics
  on top.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). UROM-specific debug (UROM Service not
  reachable, host library + DPU service version skew, RDMA
  transport failure underneath, operation type unsupported on
  this device + firmware + service version) overlays on top.

The **DOCA UROM Service** running on the DPU side is deliberately
**not in scope** — a separate artifact with its own public guide
via
`doca-public-knowledge-map ## DOCA services` (`doca-public-knowledge-map/SKILL.md#doca-services`).
Conflating the library and the service is the single most common
UROM first-app design error: surface the split explicitly whenever
a question straddles the two.
