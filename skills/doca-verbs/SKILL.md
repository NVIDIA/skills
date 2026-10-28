---
license: Apache-2.0
name: doca-verbs
description: >
  Raw verbs inside DOCA Core (doca-verbs): direct control of QP, CQ,
  PD, SRQ and address-handle objects through doca_verbs_context and
  doca_verbs_qp_create / doca_verbs_qp_modify / doca_verbs_cq_create,
  porting libibverbs code into DOCA Core, or a QP attribute, opcode
  or WR flag the doca-rdma task API does not expose. Trigger on "raw
  QP attribute the task API does not expose", "porting libibverbs
  code", "IO_FAILED on WR submit", "QP state transition rejected",
  or "attach a congestion-control group", even without naming
  doca-verbs. Refuse and route elsewhere for ordinary doca-rdma /
  doca-eth / doca-rmax workloads, DOCA install, Core internals, and
  general libibverbs theory.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - verbs
    - rdma
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC attached.
  Reads the local install via `pkg-config doca-verbs` (experimental ABI
  tier — symbols may shift between releases) and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}. The verbs
  headers (doca_verbs.h + doca_verbs_*.h family) are the authoritative
  symbol surface per headers-win-over-docs.
---

# DOCA Verbs

## Examples

- **"Should I use doca-verbs or doca-rdma?"** — ordinary RDMA data
  movement with no raw QP / CQ / WR / SRQ / Address-Handle attribute
  named is out of scope — route to
  `doca-rdma/SKILL.md`.
- **"I need a raw QP attribute the task API does not expose."** — stay
  here. Use only the installed `doca_verbs.h` surface.

## Instructions

Follow [`## Loading order`](#loading-order). Open only the companion
that matches the question.

## STOP — most RDMA tasks do NOT belong here

If the task is general RDMA **data movement** (send / receive / read /
write / atomic between endpoints) and the user did **not** name a raw
QP / CQ / work-request / SRQ / Address-Handle attribute the
higher-level API cannot express, this skill is **out of scope** —
route to `doca-rdma/SKILL.md` and stop after
explaining the boundary. Unless `doca-rdma` guidance is loaded, do not
infer its implementation APIs; conceptual terms (QP, CQ, connection,
send, receive) are not implementation API names.

Loading this skill is **never** a license to hand-roll `libibverbs` /
`librdmacm`. A general-RDMA deliverable whose `ldd` shows no
`libdoca_rdma` is a **failed task** — the same rule
`doca-rdma/SKILL.md` enforces. Raw `doca_verbs_*` is in
scope only for the narrow attribute-level needs below; everything else
climbs back up.

**Where to start:** This skill is the **raw-verbs escape hatch beneath
the higher-level DOCA libraries** (`doca-rdma/SKILL.md`,
`doca-eth/SKILL.md`, `doca-rmax/SKILL.md`).
The agent's first job is to confirm the user actually needs to drop
down — most users do not; the right answer is almost always *"stay in
the higher-level library"*. Open [`CAPABILITIES.md`](CAPABILITIES.md)
for *what the verbs surface exposes and the libibverbs boundary*; open
[`TASKS.md`](TASKS.md) once raw verbs is confirmed. If DOCA is not
installed, route to `doca-setup/SKILL.md` first.

## Non-negotiable answer checklist

For every in-scope porting, lifecycle, or debugging answer, do not declare
it complete until it explicitly states all applicable items below:

1. `doca_verbs_context` is the DOCA Core context owning the verbs
   surface; it wraps the `doca_dev`, integrates with the progress
   engine, and owns the per-object PD / MR / CQ / QP lifecycle in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
2. A submit-time success followed by `DOCA_ERROR_IO_FAILED` is
   diagnosed from the completion-queue entry's error field — not the
   submit return — before recommending retries or code changes.
3. The answer selects exactly one completion path: progress engine,
   manual CQ polling, or completion-channel delivery.
4. Both version axes are stated before any symbol is promised:
   `pkg-config --modversion doca-verbs` must agree with
   `doca_caps --version` and `pkg-config --modversion doca-common`;
   and the `doca_verbs_*` ABI ships in the EXPERIMENTAL tier — symbols
   may rename or graduate between releases and the installed headers
   outrank documentation (`doca-version/SKILL.md`).
   Device support for a verb is answered by the `doca_devinfo`
   capability query, not by either version.
5. Existing `ibv_*` handles are replaced, never mixed with
   `doca_verbs_*` handles on the same hardware objects.

## The decision this skill exists to gate

The decision every conversation loading this skill must make FIRST,
before any code-level discussion:

1. **Has the user confirmed that the matching higher-level DOCA
   library does not expose the semantic they need?** If no — route
   back: `doca-rdma/SKILL.md` for general RDMA,
   `doca-eth/SKILL.md` for Ethernet TX / RX,
   `doca-rmax/SKILL.md` for timing-precise media.
   The most common failure is recommending raw verbs *unnecessarily*
   because the user said "verbs" or "QP" without confirming the
   higher-level surface already covers the case.
2. **Is the needed semantic a specific verb / opcode / WR flag / QP
   attribute / SRQ option the higher-level library does not expose?**
   Examples: a raw WR flag `doca_rdma_task_*` does not surface; custom
   CQ handling beyond the progress engine; an esoteric QP attribute
   (path MTU, PSN, ECE); explicit SRQ control; congestion-control
   group (`doca_verbs_cc_group_*`) attachment; Address-Handle tuning
   (DGID / DLID / SL / SGID index / hop limit / traffic class / UDP
   source port). If yes — in scope.
3. **Is the user porting existing libibverbs code into the DOCA Core
   model?** In scope, AND the agent must teach the porting path
   (replace `ibv_*` handles with `doca_verbs_*` handles, integrate via
   `doca_verbs_context_create`, drive completions via the DOCA
   progress engine instead of polling the CQ) — not a mechanical 1:1
   replacement.

If none of (1)-(3) apply, the answer to *"should I use `doca-verbs`?"*
is **no** — route back to the matching higher-level library. A raw-verbs
skill that talks the user *out* of raw verbs is doing its job.

## Audience

This skill serves **external developers building applications that
consume the DOCA Verbs library** — code calling `doca_verbs_*`
(directly in C/C++, or via FFI/bindings) for raw QP / CQ / PD / MR /
SRQ / Address-Handle / Ethernet-SQ / Ethernet-RQ control inside a
DOCA Core context. It is *not* for NVIDIA developers contributing to
DOCA Verbs, and *not* the entry point for general DOCA RDMA / Eth /
RMAX work — that belongs in the matching higher-level library skill.

## Language scope

DOCA Verbs ships as a C library (`pkg-config` module `doca-verbs`);
public headers live under the installed DOCA tree
(`$(pkg-config --variable=includedir doca-common)/doca_verbs.h` plus
the `doca_verbs_*.h` family) and, per the headers-win-over-docs rule
in `doca-version/SKILL.md`, are the
authoritative symbol surface. C / C++ is the canonical case `TASKS.md`
assumes; other languages reach the same `*.so` through FFI, and the
skill keeps its rules language-neutral while routing to the public C
ABI as the authoritative surface.

## When to load this skill

Load this skill ONLY after the user has confirmed the matching
higher-level DOCA library does not expose the needed semantic.
Concretely:

- The user asks *"do I need to drop to `doca-verbs` for this?"* —
  load to answer, but expect *"no, stay in the higher-level
  library"* unless the user can name the verb / opcode / option the
  higher-level library does not surface.
- The user wants a specific raw WR flag, QP option, SRQ feature,
  custom CQ-handling pattern, or congestion-control group attachment
  the higher-level library does not expose.
- The user is porting libibverbs code into the DOCA Core model and
  needs the integration path (lifecycle, progress engine, no-mixing).
- A `DOCA_ERROR_*` from a `doca_verbs_*` call needs diagnosis —
  including IO_FAILED, where the answer lives on the
  completion-queue entry, not the submit return.
- Designing non-C bindings wrapping the verbs C ABI that must honor
  the boundary, lifecycle, and cap-query rules.


Do **not** load for: general RDMA (`doca-rdma/SKILL.md`);
Ethernet queueing (`doca-eth/SKILL.md`); timing-precise
media (`doca-rmax/SKILL.md`); cases another
higher-level library covers (`doca-flow/SKILL.md` for
steering, `doca-sta/SKILL.md` for NVMe-oF target acceleration); DOCA install
(`doca-setup/SKILL.md`); or general orientation —
all routed via
`doca-public-knowledge-map/SKILL.md`.

## What this skill provides

This is a **thin loader**. The substantive raw-verbs material lives
in two companion files:

- `CAPABILITIES.md` — what `doca-verbs` can express on this
  version: the higher-level-vs-verbs selection table, the libibverbs
  boundary, the verbs object model (`doca_verbs_context` + QP / CQ /
  PD / MR / SRQ / Address-Handle / Completion-Channel / Ethernet-SQ /
  Ethernet-RQ / CC-group), the capability-query surface
  (`doca_verbs_query_device` + `doca_verbs_device_attr_get_*`), the
  error taxonomy (mapped onto `DOCA_ERROR_*` with the IO_FAILED →
  completion-entry overlay), the observability surface (progress
  engine vs manual CQ polling vs comp-channel delivery), and the
  safety policy gating the no-mixing-with-libibverbs rule.
- `TASKS.md` — workflows for the six in-scope verbs: `configure`,
  `build`, `modify`, `run`, `test`, `debug`, plus a `Deferred task
  verbs` block. Every workflow assumes the drop-down decision above
  was made; `## configure` re-confirms it.

The skill assumes DOCA is installed and the user has the privileges
their install profile expects (RDMA stack on host, same as
`doca-rdma/SKILL.md`). It does not cover installing
DOCA — route to `doca-setup/SKILL.md`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates bundle.
It deliberately does not contain — and pull requests should not add:

- **Pre-written DOCA Verbs application source code, in any
  language.** The verified verbs source is the shipped C samples on
  the install (via `ls /opt/mellanox/doca/samples/`); the agent
  routes there and prescribes a minimum-diff modification via the
  modify-a-sample workflow in
  `doca-programming-guide/SKILL.md`,
  layered with the verbs overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs the
  manifest *in the user's project* against the installed DOCA, where
  `pkg-config --modversion doca-verbs` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock artifact labeled "reference" reads as buildable and
  misleads.
- **A migration script from libibverbs to `doca-verbs`.** Porting is
  *judgment*, not mechanical replacement — see
  [`TASKS.md ## modify`](TASKS.md#modify).

## Loading order

1. Confirm the question is in scope by walking the drop-down decision above.
2. **For the higher-level-vs-verbs split, the libibverbs boundary,
   the object model, capability discovery, error taxonomy,
   observability, and safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other, to the higher-level
libraries (`doca-rdma/SKILL.md`,
`doca-eth/SKILL.md`,
`doca-rmax/SKILL.md`) as climb-back homes, to
`doca-common/SKILL.md` for the `doca_dev` /
`doca_pe` / `doca_ctx` foundation primitives, to
`doca-version/SKILL.md` for canonical version
rules, and to
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public docs or the
install layout".

## Related skills

When a step below routes to another skill that is not already loaded,
read that skill first; if unavailable, stop and name it and the layer it
covers — do not continue that layer from memory.

- `doca-rdma/SKILL.md` — the canonical higher-level
  RDMA library and the home this skill routes most RDMA users *back*
  to. Any conversation loading `doca-verbs` for an RDMA question
  should also load `doca-rdma` so the climb-back answer is immediate.
- `doca-eth/SKILL.md` — the higher-level Ethernet
  queue library. `doca-verbs` also exposes Ethernet SQ / RQ verbs
  (`doca_verbs_eth_sq_*`, `doca_verbs_eth_rq_*`); once the user
  confirms `doca-eth` lacks the needed option (TS-source-type tuning,
  plane-index pinning, multi-pkt-send-WQE), this skill takes over.
- `doca-rmax/SKILL.md` — the higher-level Rivermax
  library for timing-precise media. Most Rivermax cases stay there;
  raw verbs is the escape hatch for the rare case needing a verb
  Rivermax does not expose.
- `doca-common/SKILL.md` — the foundation every
  DOCA context (including `doca_verbs_context`) rests on:
  `doca_dev` / `doca_pe` / `doca_ctx`, the `doca_devinfo` cap-query
  rule, and the lifecycle. This skill layers verbs specifics on top.
- `doca-public-knowledge-map/SKILL.md` —
  the routing table for public DOCA docs and the install layout; the
  DOCA Verbs public guide is listed there.
- `doca-setup/SKILL.md` — env preparation,
  install verification, and the *no install yet* path via the NGC
  DOCA container. This skill assumes its preconditions hold.
- `doca-version/SKILL.md` — canonical DOCA
  version-handling rules. `## Version compatibility` cross-links the
  four-way match rule (with `doca-verbs.pc` joining the match set)
  and the cap-query-is-runtime-authority rule.
- `doca-structured-tools-contract/SKILL.md` —
  the bundle's structured-tools precedence rule (detect / prefer /
  fall back / report). The Command appendix in [TASKS.md](TASKS.md)
  honors it.
- `doca-programming-guide/SKILL.md` —
  general DOCA programming patterns shared by every library:
  `pkg-config` + meson build, modify-a-shipped-sample first-app, the
  universal lifecycle, the `DOCA_ERROR_*` taxonomy, and the debug
  order. This skill layers raw-verbs specifics on top.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Raw-verbs debug (completion-entry inspection,
  no-mixing-with-libibverbs, lifecycle in verbs terms) overlays on top.
- `doca-hardware-safety/SKILL.md` —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays.
