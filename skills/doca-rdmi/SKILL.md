---
license: Apache-2.0
name: doca-rdmi
description: >
  Use this skill when the user is doing hands-on DOCA RDMI (RDMA
  Initiator) programming — picking doca-rdmi vs doca-rdma for an
  accelerator-initiated one-sided RDMA flow, standing up a
  doca_rdmi_connection or doca_rdmi_poster, attaching a
  doca_dpa_completion or doca_verbs_cq before doca_ctx_start(),
  retrieving the DPA-side handle for a DPA kernel, auditing whether
  a doca_rdmi_* symbol is EXPERIMENTAL on this DOCA, or debugging
  DOCA_ERROR_* returns from RDMI calls. Trigger even when the user
  does not say "DOCA RDMI" or "initiator" — implicit phrasings
  include "my DPA kernel needs to post RDMA writes to a remote
  responder", "DPA kernel sees no completions", "function not found
  on doca_rdmi_* at link time", "DOCA_ERROR_BAD_STATE from
  completion attach", or "the DPA posted but the work request never
  arrived". Refuse and route elsewhere for two-sided or host-CPU
  RDMA, the DPA programming model, GPU-side RDMA initiation, or
  general RDMA/IB/RoCE concepts — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached, plus the DOCA DPA toolchain and a
  DPA-capable BlueField when the data path runs on the DPA. Reads
  the user's local install via `pkg-config doca-rdmi` (alongside
  doca-common, doca-verbs, doca-dpa) and inspects
  /opt/mellanox/doca/{lib,include,samples}.
---

# DOCA RDMA Initiator

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on RDMI work** on a host or BlueField with
the DOCA package set that ships the `doca-rdmi` library. Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something
(install / configure / build / modify / run / test / debug / use);
open [`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what can RDMI express on this version* — the object model, the
DPA-side handle types, the relationship to `doca-rdma`, the
EXPERIMENTAL-tag policy, and the safety overlay.

**End-to-end "walk me through doca-rdmi" questions are answerable
entirely from this skill.** Go straight to
[`TASKS.md ## end-to-end (quickref)`](TASKS.md#end-to-end-quickref),
which carries the self-contained install-check → device/cap
discovery → sample → `pkg-config` build → run → debug walkthrough
with the exact commands. You do **not** need to open `doca-setup` or
`doca-programming-guide` to answer an RDMI build/run/debug question.
Route to [`doca-setup`](../../doca-setup/SKILL.md) **only** when the
user has no DOCA install at all.

## Example questions this skill answers well

The CLASSES of RDMI questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"Should I use `doca-rdmi` or `doca-rdma` for this case?"** —
  worked example: *"I have a DPA kernel that needs to fire 1 MB
  RDMA writes at a remote responder; which library?"*. Answered by
  the *initiator-side vs general-purpose* selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  surface-selection table + the routing back to
  [`doca-rdma`](../doca-rdma/SKILL.md) when the use case is
  two-sided or host-CPU initiated.
- **"How do I bring up an RDMI connection on the DPA datapath?"** —
  worked example: *"create a `doca_rdmi_connection`, attach a DPA
  completion context, hand the DPA-side handle to my kernel"*.
  Answered by the connection-object lifecycle in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the configure walk in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do connection and poster relate — when do I need both?"**
  — worked example: *"my application receives work requests AND
  posts RDMA writes; do I need a `doca_rdmi_connection` plus a
  `doca_rdmi_poster`, or one of them?"*. Answered by the
  two-object model in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the modify-from-sample slot table in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"How do I drive completions on the DPA side?"** — worked
  example: *"hook the connection to a `doca_dpa_completion` so my
  kernel polls completions directly"*. Answered by the DPA-side
  completion-attach pattern in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the run-side wiring in
  [`TASKS.md ## run`](TASKS.md#run), cross-linked into
  [`doca-dpa`](../doca-dpa/SKILL.md) for the DPA programming
  surface itself.
- **"Is the symbol I want available — and is it stable enough to
  ship?"** — worked example: *"is `doca_rdmi_poster_post` GA on
  my installed DOCA, or still EXPERIMENTAL?"*. Answered by the
  EXPERIMENTAL-tag policy in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the version-discovery rule
  (`pkg-config --modversion doca-rdmi`) pinned in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"What does this `DOCA_ERROR_*` from a `doca_rdmi_*` call
  mean?"** — worked example: *"`DOCA_ERROR_BAD_STATE` from
  `doca_rdmi_connection_dpa_completion_attach`"*. Answered by the
  RDMI overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in [`TASKS.md ## debug`](TASKS.md#debug)
  that escalates to [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building DPA / GPU-resident
DOCA applications that need to *initiate* one-sided RDMA operations
against a remote responder** — i.e., users whose accelerator-side
code wants to post sends, writes, or reads directly from the
accelerator without round-tripping through the host CPU. The
canonical caller is a DPA kernel that has been compiled with
`doca-dpacc-compiler` and runs on the BlueField DPA datapath; a
GPU-side caller that drives the DPU's RDMA queues is the
sister case covered by [`doca-gpi`](../doca-gpi/SKILL.md). This
skill is *not* for NVIDIA developers contributing to DOCA RDMI
itself, and it is not the right surface for general host-CPU
two-sided RDMA — that belongs to
[`doca-rdma`](../doca-rdma/SKILL.md).

## Language scope

DOCA RDMI ships as a C library with the `pkg-config` module name
`doca-rdmi`. The library's *host-side* surface
(`doca_rdmi_connection_*`, `doca_rdmi_poster_*`) is C; the *DPA-side*
surface that the accelerator kernel uses is also C, compiled against
the DOCA DPA toolchain documented in
[`doca-dpa`](../doca-dpa/SKILL.md). Other-language consumers (Rust,
Go, Python, …) consume the host-side `*.so` through FFI; the skill's
contribution in that case is to keep the connection / poster
lifecycle, the EXPERIMENTAL-tag policy, the DPA-side handoff rules,
and the safety overlay language-neutral, and to route the agent to
the public C ABI as the authoritative surface that any wrapper will
eventually call. The DPA-side surface is *not* wrappable in another
language — it is compiled and linked into the DPA binary itself.

## When to load this skill

Load this skill when the user is doing **hands-on DOCA RDMI work**
on a host or BlueField with DOCA installed. Concretely:

- Deciding between `doca-rdmi` and `doca-rdma` for a new
  one-sided RDMA workload that runs from the DPA datapath.
- Creating a `doca_rdmi_connection` or `doca_rdmi_poster`,
  attaching a `doca_dpa_completion` or a `doca_verbs_cq`, and
  starting the context on the DPA datapath.
- Wiring the DPA-side handle returned by
  `doca_rdmi_connection_get_dpa_handle` /
  `doca_rdmi_poster_get_dpa_handle` into a DPA kernel that calls
  the matching device-side header
  (`doca_rdmi_dev_connection.h`, `doca_rdmi_dev_poster.h`,
  `doca_rdmi_dev_cqe.h`).
- Acknowledging receive completions on the host side via
  `doca_rdmi_connection_recv_ack` after the DPA kernel consumed
  them.
- Auditing which RDMI symbols are EXPERIMENTAL on the installed
  DOCA version, before declaring an RDMI-using component
  production-stable.
- Debugging a `DOCA_ERROR_*` returned by a `doca_rdmi_*` call and
  deciding whether the cause is a configuration mistake, a
  lifecycle ordering bug, an unsupported capability on this device,
  or a layer below DOCA.

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, or two-sided host-CPU RDMA questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-setup`](../../doca-setup/SKILL.md), and
[`doca-rdma`](../doca-rdma/SKILL.md) respectively.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive RDMI-specific
material lives in two companion files:

- `CAPABILITIES.md` — what RDMI can express on this version: the
  `doca_rdmi_connection` and `doca_rdmi_poster` object model, the
  DPA-side completion-attach pattern, the relationship to
  `doca-verbs` (RDMI builds on a `doca_verbs_context`) and to
  `doca-dpa` (the accelerator-side datapath), the EXPERIMENTAL-tag
  rule for version handling, the RDMI overlay on the cross-library
  `DOCA_ERROR_*` taxonomy, the observability surface (completion
  events on the PE / DPA-side completions), and the safety policy
  that gates posting work from an accelerator kernel into a remote
  responder's memory.
- `TASKS.md` — step-by-step workflows for the eight in-scope verbs:
  `install`, `configure`, `build`, `modify`, `run`, `test`,
  `debug`, `use`. Plus a `## rollback` overlay (RDMI-specific
  five-step teardown for the verbs / connection / DPA-attach /
  MR stack) and the 5-phase universal debug-loop instantiation
  appended to `## debug`. Plus a `Deferred task verbs` block
  that points out-of-scope questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA itself — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA RDMI application source code, in any
  language.** The agent's job is to route the user to verified
  reference code on the user's installed DOCA and to prescribe a
  minimum-diff modification via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the RDMI-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify). Because every RDMI
  symbol is EXPERIMENTAL at the time of writing, the skill
  refuses to author RDMI source code from documentation prose —
  the API can change between releases and the resulting code may
  not even compile.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against the
  user's installed DOCA, where `pkg-config --modversion doca-rdmi`
  is the source of truth.
- **DPA-side kernel templates.** The DPA-side surface is owned by
  [`doca-dpa`](../doca-dpa/SKILL.md); RDMI's DPA-side headers are
  *consumed by* the DPA programming model documented there. This
  skill names the RDMI-specific handoff (the DPA handle type, the
  completion-attach call) but does not author DPA kernels.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the RDMI object model, the DPA-side handoff pattern, the
   EXPERIMENTAL-tag policy, the error taxonomy, observability, and
   safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — install, configure, build,
   modify, run, test, debug, use — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other and to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "RDMI-specific guidance".

## Related skills

- [`doca-rdma`](../doca-rdma/SKILL.md) — the higher-level RDMA
  library covering two-sided and host-CPU-initiated RDMA. RDMI is
  the focused initiator-side surface; doca-rdma is the right
  answer for the majority of RDMA use cases. The selection table
  in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  is the load-bearing decision aid.
- [`doca-dpa`](../doca-dpa/SKILL.md) — the DOCA DPA programming
  surface. RDMI returns DPA-side handles
  (`doca_dpa_dev_rdmi_connection_t`, `doca_rdmi_dev_poster_t`)
  that the DPA kernel uses through the device-side headers
  (`doca_rdmi_dev_connection.h`, `doca_rdmi_dev_poster.h`,
  `doca_rdmi_dev_cqe.h`); the DPA toolchain, kernel build, and
  execution model are owned by that skill.
- [`doca-gpi`](../doca-gpi/SKILL.md) — the GPU-side sister of
  this skill. GPI is the channel/queue surface a CUDA kernel
  uses to initiate RDMA; RDMI is the DPA-side surface. Both
  layer on the same DOCA RDMA / verbs substrate; either may
  apply depending on whether the initiator is on the DPA or on
  the GPU.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — the
  routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  Core-context lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy. This skill layers RDMI specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). RDMI-specific debug overlays on top of it.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the bundle-wide hardware-safety meta-policy. The `## Safety
  policy` overlay in `CAPABILITIES.md` cross-links it.
- [`doca-version`](../../doca-version/SKILL.md) — the version
  detection / four-way match rule every per-artifact `##
  Version compatibility` anchor builds on. This skill quotes
  the RDMI-specific overlay only.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the JSON-schema contracts for the agent-preferred structured
  helpers (env probe, capability snapshot, version-matrix
  lookup); the `## Command appendix` in `TASKS.md` defers to
  them before falling back to the manual chain.

