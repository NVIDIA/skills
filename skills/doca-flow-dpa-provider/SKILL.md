---
license: Apache-2.0
name: doca-flow-dpa-provider
description: >
  Use this skill when the user is doing hands-on DOCA Flow DPA
  Provider work — exporting a `doca-flow` pipe or external
  resource (index-selector/memory) into BlueField DPA
  address space so a DPACC-built kernel can read counters, mutate
  hash-pipe entries, and update/read memory or index-selector
  resources inline with Flow. Covers per-port
  `doca_flow_dpa_ctx`, three queue types
  (general/resources-write/resources-read), the order-sensitive
  export handshake (`_export_prepare` → add entries → `_export` →
  `_get_device_addr`), and DPA-side device API. Trigger even
  when the user does not say "DOCA Flow DPA Provider" — implicit
  phrasings include "DPA kernel never sees entries in the exported
  pipe", "BAD_STATE from `_pipe_export`", "how do I disable a hash
  entry from a DPA kernel", "DPA memory read returns no value", or
  "DPA-side post keeps returning AGAIN". Refuse and route
  elsewhere for `doca-flow` pipe construction, generic
  host-side DPA (`doca-dpa`), or DPA-side kernel-writing — those
  belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU exposing
  a DPA processor visible to the host, PLUS the DPACC compiler
  installed at a version matched to DOCA per the DOCA Compatibility
  Policy. Reads the user's local install via `pkg-config
  doca-flow-dpa-provider` (cross-checked with `doca-flow`,
  `doca-dpa`, and installed `dpacc`). ConnectX-only hosts cannot
  use this library.
---

# DOCA Flow DPA Provider

**Where to start:** This skill assumes DOCA is already
installed, the user has a BlueField with a DPA processor that
the host can see through DOCA, the user already programs DOCA
Flow from the host (`doca-flow`) and already runs DPA kernels
from the host (`doca-dpa` / DPACC compiler), and the user is
doing **hands-on DPA Flow Provider work** — i.e. using
`doca-flow-dpa-provider` to export an existing DOCA Flow pipe
or external resource into the DPA address space so a DPA
kernel can manipulate it inline. Open [`TASKS.md`](TASKS.md)
if the user wants to *do* something (configure / build /
modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what can the provider express* on this version + this
BlueField generation. If the user has not installed DOCA yet,
route to [`doca-setup`](../../doca-setup/SKILL.md) first; if
the user has not stood up a host-side Flow pipe yet, route to
[`doca-flow`](../doca-flow/SKILL.md) first; if the user has
not stood up host-side DPA execution yet, route to
[`doca-dpa`](../doca-dpa/SKILL.md) first.

## Example questions this skill answers well

The CLASSES of DPA Flow Provider questions this skill is built
to answer, each with one worked example. The agent should
treat the *class* as the load-bearing piece — the worked
example is a single instance.

- **"Is this library even the right tool for what I want?"** —
  worked example: *"I want my DOCA Flow pipe to be readable
  from a DPA kernel, OR I want to do something fancier than
  the host-side `doca-flow` API exposes"*. Answered by the
  decision rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  ("three-program model: when this library is the right
  surface vs pure host-side Flow vs pure DPA"). Most
  first-time askers do NOT need this library; the skill's
  first job is to confirm they do.
- **"How do I export an existing DOCA Flow pipe to the DPA
  side?"** — worked example: *"I have a Flow pipe with a
  counter on every entry; I want a DPA kernel to read those
  counters inline and decide what to do next"*. Answered by
  the host-side export sequence in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the bring-up steps in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I drive the exported pipe from inside the DPA
  kernel?"** — worked example: *"I have the device address of
  an exported hash pipe; how do I disable an entry from the
  DPA, and how do I know the operation completed?"*.
  Answered by the device-side API surface in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  ("DPA-side consumption") + the host-side queue allocation
  + DPA-side polling step in
  [`TASKS.md ## run`](TASKS.md#run).
- **"What does this `DOCA_ERROR_*` from a
  `doca_flow_dpa_*` call mean, and is the bug on the host
  side, on the DPA side, or in the export handshake between
  them?"** — worked example: *"`DOCA_ERROR_BAD_STATE` from
  `doca_flow_dpa_pipe_export`"*. Answered by the provider
  overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).
- **"Why did my pipe export *succeed* but my DPA kernel never
  sees entries?"** — worked example: *"I called
  `doca_flow_dpa_pipe_export_prepare` AFTER adding entries to
  the pipe"*. Answered by the lifecycle ordering rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the staged-export workflow in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Is the host-side / DPA-side provider API I'm reading
  about on my installed DOCA?"** — worked example: *"is the
  three-queue-type `doca_flow_dpa_queues_create` available
  against the DOCA + DPACC versions on this host, or am I
  still on the older `doca_flow_dpa_pipe_queues`-based
  surface?"*. Answered by the version-compatibility overlay
  in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  provider-specific overlay inherited from
  [`doca-dpa`](../doca-dpa/SKILL.md).

## Audience

This skill serves **external developers building applications
that consume the DOCA Flow DPA Provider library** — i.e.,
users who already have a host-side `doca-flow` pipe and a
host-side `doca-dpa` execution context, and who want to wire
the two together so a DPA kernel can read counters from / mutate
entries of / read or write external resources tied to that
Flow pipe inline, instead of round-tripping through the host
CPU. It is *not* for NVIDIA developers contributing to the
provider library itself, nor is it for users who only need
host-side Flow programming (that is `doca-flow`) or who only
need generic DPA compute that has nothing to do with Flow
(that is `doca-dpa`).

**Language scope.** DOCA Flow DPA Provider ships as a *paired*
library: a host-side C library (pkg-config module
`doca-flow-dpa-provider`, header
`doca_flow_dpa_provider.h`) plus a DPA-side device library
that the DPACC compiler links into the DPA-side translation
unit when the kernel includes `doca_flow_dpa_provider_dev.h`.
Both halves are C. The shipped samples (under the installed
DOCA samples tree) include both translation units. Other-
language host-side consumers can FFI the host C library, but
the DPA-side kernel has no FFI escape hatch (the DPACC
compiler accepts a single translation unit per kernel image).
The skill keeps the lifecycle, capability-discovery, env-
precondition, and error-taxonomy guidance language-neutral on
the host side, and points all DPA-side kernel-writing
questions at the public DOCA DPA and DOCA Flow programming
guides via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## When to load this skill

Load this skill when the user is doing hands-on DOCA Flow DPA
Provider work, in any host language plus a DPA-side
translation unit built by `dpacc`. Concretely:

- Initializing a `doca_flow_dpa_ctx` against a `flexio_process`
  that maps to a BlueField with a DPA processor visible to
  the host AND a host-side `doca-flow` port already brought
  up against the same device.
- Allocating the three DPA-side queue types (general /
  entries, resources-write, resources-read) for a port via
  `doca_flow_dpa_queues_create`.
- Exporting a host-side `doca_flow_pipe` to the DPA address
  space via the
  `doca_flow_dpa_pipe_export_prepare` → `_pipe_export` →
  `_pipe_get_device_addr` sequence, BEFORE any entry is added
  to the pipe.
- Exporting an external Flow resource (index-selector
  resource, memory resource) to the DPA via the matching
  `doca_flow_dpa_external_resource_*` family.
- Handing the resulting `doca_flow_dpa_addr` device pointer to
  the DPA-side kernel so the kernel can call
  `doca_flow_pipe_hash_*` / `doca_flow_external_resource_*`
  / `doca_flow_queue_poll_completion` on it.
- Debugging a `DOCA_ERROR_*` from a `doca_flow_dpa_*` call —
  in particular disambiguating *export called too late in the
  pipe lifecycle* from *queues not yet created* from *DPA-
  side queue full and not drained*.
- Designing host-side bindings in a non-C language that drive
  the provider's host-side setup and hand the device address
  off to a DPA application image built separately with
  `dpacc`.

Do **not** load this skill for general DOCA Flow programming
(use [`doca-flow`](../doca-flow/SKILL.md)), generic host-side
DPA work (use [`doca-dpa`](../doca-dpa/SKILL.md)), or DPA-
side kernel-writing itself (route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA DPA / DPACC / Flow programming guides).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
provider-specific material lives in two companion files:

- `CAPABILITIES.md` — what the host-side and DPA-side provider
  API can express on this version + this BlueField
  generation: the per-port `doca_flow_dpa_ctx` context, the
  three queue types (`DOCA_FLOW_DPA_QUEUE_TYPE_GENERAL` /
  `_RESOURCES_WRITE` / `_RESOURCES_READ`), the
  pipe-export handshake (prepare → export → get-device-addr),
  the parallel external-resource-export handshake, the DPA-
  side device API for hash-pipe entry manipulation
  (`doca_flow_pipe_hash_enable_index` /
  `_hash_disable_index` / `_hash_replace_index`), the DPA-
  side device API for external resources
  (`doca_flow_external_resource_index_selector_modify*`,
  `doca_flow_external_resource_memory_update`,
  `doca_flow_external_resource_memory_read*`), the
  completion-queue polling
  (`doca_flow_queue_poll_completion`), the
  three-program-model rule (host-side `doca-flow` + host-side
  `doca-dpa` + DPA-side kernel), the error taxonomy mapped
  onto the cross-library `DOCA_ERROR_*` set, the
  observability surface, and the safety policy that gates the
  export lifecycle.
- `TASKS.md` — step-by-step workflows for the in-scope
  provider verbs: `install`, `configure`, `build`, `modify`,
  `run`, `test`, `debug`, `use`. Plus a `Deferred task verbs`
  block that points out-of-scope questions at the right next
  skill.

The skill assumes a host where DOCA is already installed at
the standard location, a BlueField with a DPA processor is
physically present and visible to the host, the DPACC
compiler is installed at a version matched to the DOCA
install per the DOCA Compatibility Policy, the user already
knows how to bring up a `doca-flow` port and create a
host-side `doca_dpa` (or its FlexIO equivalent) for kernel
execution, and the user has at least a sketch of the DPA-side
kernel that will consume the exported pipe device address.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Flow DPA Provider application source
  code or DPA-side kernel source, in any language.** The
  verified provider source is the shipped C + DPA-side
  samples under the installed DOCA samples tree (route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the per-install sample tree path). The agent's job is
  to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the provider-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, …) parked inside the skill. The agent
  constructs the build manifest *in the user's project
  directory* against the user's installed DOCA + DPACC
  compiler, where `pkg-config --modversion
  doca-flow-dpa-provider` (alongside `doca-flow` and
  `doca-dpa`) and the installed `dpacc` are the joint sources
  of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of
  any kind. A mock or incomplete artifact in this skill's
  tree, even one labeled "reference", is misleading: users
  will read it as buildable.
- **Host-side `doca-flow` pipe-spec content.** That is
  [`doca-flow`](../doca-flow/SKILL.md)'s scope; this skill
  *exports* an already-constructed pipe and does not redefine
  how to construct it.
- **DPA-side kernel programming generalities.** Writing the
  DPA-side function body, allocating DPA memory inside the
  kernel, intrinsics, DPA-side libraries like
  `doca-dpa-comms` and `doca-dpa-verbs` — out of scope. Route
  to the public DOCA DPA / DPACC guides via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (Flow-pipe-exported-to-DPA work, not pure
   `doca-flow` work and not generic `doca-dpa` work).
2. **For the provider capability matrix, the three-program
   model, the per-port `doca_flow_dpa_ctx`, the three queue
   types, the pipe-export and external-resource-export
   handshakes, the DPA-side device API surface, the
   completion model, the error taxonomy, the observability
   surface, and the safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — install, configure, build,
   modify, run, test, debug, use — see
   [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-flow`](../doca-flow/SKILL.md) and
[`doca-dpa`](../doca-dpa/SKILL.md) for the host-side
surfaces this library bridges,
[`doca-version`](../../doca-version/SKILL.md) for the
canonical DOCA version-handling rules (with the
DOCA-and-DPACC overlay inherited from
[`doca-dpa`](../doca-dpa/SKILL.md)), and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
Flow / DOCA DPA / DPACC programming guides, or in the on-disk
install layout" rather than "provider-specific guidance".

## Related skills

- [`doca-flow`](../doca-flow/SKILL.md) — the canonical
  host-side Flow programming skill. The Flow pipe this
  library exports MUST be brought up against
  [`doca-flow CAPABILITIES.md ## Capabilities and modes`](../doca-flow/CAPABILITIES.md#capabilities-and-modes)
  first; this skill only adds the DPA-export layer on top.
- [`doca-dpa`](../doca-dpa/SKILL.md) — the host-side DPA
  control library. The DPA kernel that consumes the
  exported device address is loaded and launched through
  `doca-dpa` (or its FlexIO-process equivalent); this skill
  inherits the two-side-program model, the DPACC-and-DOCA
  version-match rule, and the env-precondition matrix from
  there.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing table for every public DOCA documentation source
  (DOCA Flow guide at
  <https://docs.nvidia.com/doca/sdk/doca-flow/index.html>;
  DOCA DPA guide at
  <https://docs.nvidia.com/doca/sdk/doca-dpa/index.html>;
  DPACC compiler guide and DPA Tools umbrella reachable from
  the same routing table) and the on-disk layout of an
  installed DOCA package.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, DPACC compiler install /
  verification, and the *I have no install yet* path with
  the public NGC DOCA container. This skill assumes its
  preconditions are satisfied AND that DPACC is installed at
  a version that matches DOCA.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the provider-specific *DOCA-and-DPACC must match* overlay
  per the DOCA Compatibility Policy (inherited from
  [`doca-dpa`](../doca-dpa/SKILL.md)).
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns: the canonical
  `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  Core-context lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill
  layers provider specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Provider-specific debug (lifecycle-
  ordering between Flow pipe creation, queue creation, export
  prepare, entry add, export; queue-full on DPA-side polling;
  pipe device address handed to a kernel that targets a
  different `doca_flow_dpa_ctx`) overlays on top.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  cross-cutting hardware-safety meta-policy that this
  skill's `## Safety policy` overlays. Because the exported
  pipe is mutated inline by DPA-side kernel code, the
  validate-before-commit discipline from `doca-flow` plus the
  two-side-program rebuild discipline from `doca-dpa` BOTH
  apply, and the meta-policy provides the cross-cutting
  framing.

