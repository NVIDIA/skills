---
license: Apache-2.0
name: doca-pcc
description: >
  Use this skill when the user is doing hands-on host-side DOCA
  PCC work to load a CUSTOM Programmable Congestion Control
  algorithm onto a BlueField DPU — creating per-port `doca_pcc`
  contexts, loading a `dpacc`-compiled `doca_pcc_app` onto the
  `doca_dev` for the RoCE-bearing port, parameterizing it,
  walking triple-axis capability discovery (DOCA cap-query +
  DPA-capable BlueField + firmware custom-PCC slot enabled), or
  debugging `DOCA_ERROR_*` from `doca_pcc_*`. Trigger even
  without explicit "DOCA PCC" phrasing — implicit forms include
  "loading my own congestion control onto a BF port",
  "DOCA_ERROR_NOT_PERMITTED on algorithm load",
  "DOCA_ERROR_DRIVER when I attach my custom algorithm", "my
  custom rate-update isn't affecting RoCE traffic", or "load
  succeeds but no on-wire change". Refuse and route elsewhere
  for DPA-side algorithm-body design, the `pcc_counters`
  CLI, default factory PCC in ConnectX firmware, or setting up
  the RDMA / RoCE traffic — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU whose
  DPA processor is exposed to the host AND whose firmware has
  the custom-PCC slot enabled. Also requires the DPACC compiler
  installed at a version matched to DOCA per the DOCA
  Compatibility Policy. Reads the user's local install via
  `pkg-config doca-pcc` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA PCC

**Where to start:** This skill assumes DOCA is already installed,
the user's BlueField has a DPA processor that the host can see
through DOCA, the BlueField firmware has the custom-PCC slot
enabled, and the user is doing **hands-on custom PCC work from
the host side** — i.e. using `doca-pcc` to load a DPA-side
congestion control algorithm onto the BlueField, attach it to a
port handling RDMA / RoCE traffic, and parameterize it from the
host. Open [`TASKS.md`](TASKS.md) if the user wants to *do*
something (configure / build / modify / run / test / debug);
open [`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what can the host-side PCC API express* on this version + this
BlueField generation + this firmware. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user is
asking how to *write* the DPA-side congestion-control algorithm
itself (the code that runs on the DPA processor, compiled by
`dpacc`), that is a different scope — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA PCC programming guide and to
[`doca-dpa`](../doca-dpa/SKILL.md) for the host-side DPA
lifecycle this skill builds on. If the user only wants to
*inspect* PCC counters at runtime without writing a custom
algorithm, that is the `pcc_counters` CLI tool — route via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools);
this skill is for *custom* algorithms only.

## Example questions this skill answers well

The CLASSES of PCC questions this skill is built to answer,
each with one worked example. The agent should treat the *class*
as the load-bearing piece — the worked example is a single
instance.

- **"How do I deploy my own custom congestion control algorithm
  onto a BlueField port carrying RDMA / RoCE traffic?"** —
  worked example: *"load a small DPA-side PCC algorithm and
  attach it to the BlueField port that handles my RoCE traffic"*.
  Answered by the two-side-program model + the host-side
  load-and-attach workflow in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the bring-up steps in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Does this BlueField + firmware actually allow a custom PCC
  algorithm, and which PCC features does my DOCA install
  expose?"** — worked example: *"my host has a BlueField and the
  default factory PCC works; can I drop in a custom algorithm
  instead?"*. Answered by the triple-axis precondition rule
  (BlueField generation must carry a DPA, firmware must have the
  custom-PCC slot enabled, `doca_pcc_cap_*` against the active
  `doca_devinfo` must agree) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the env-precondition checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"Why does my custom PCC fail with `DOCA_ERROR_NOT_PERMITTED`
  even though I have `doca_dev` access?"** — worked example:
  *"the BlueField firmware in this host has the custom-PCC slot
  disabled"*. Answered by the permission matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the firmware-side env fix in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"Is this `doca-pcc` library the right tool for what I want,
  or do I want the default firmware PCC or the
  `pcc_counters` CLI?"** — worked example: *"I just want to
  read PCC counters without touching the algorithm"*. Answered
  by the path-selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the deferred-topic boundaries in
  [`CAPABILITIES.md ## Deferred topic boundaries`](CAPABILITIES.md#deferred-topic-boundaries)
  which route to
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
  for the counter tool.
- **"Is the host-side PCC API I'm reading about on my installed
  DOCA?"** — worked example: *"is the host-side load helper I
  see in the docs available against the DOCA + DPACC versions on
  this host?"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  PCC-specific *DOCA must match DPACC* overlay inherited from
  [`doca-dpa`](../doca-dpa/SKILL.md).
- **"What does this `DOCA_ERROR_*` from a `doca_pcc_*` call mean
  and which layer caused it?"** — worked example:
  *"`DOCA_ERROR_DRIVER` on the host-side algorithm-load call —
  is it DOCA, the firmware-side custom-PCC slot, or the DPACC-
  produced image?"*. Answered by the PCC overlay on the
  cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications
that consume the DOCA PCC library from the host side** — i.e.,
users whose code calls `doca_pcc_*` from host C / C++ to stand
up the per-PCC-instance context, load a DPA-side PCC algorithm
image that `dpacc` produced from their DPA-side source, attach
it to the BlueField port that carries the RDMA / RoCE traffic
the algorithm is meant to control, parameterize the algorithm,
start the context, and observe runtime reports back from the
algorithm. It is *not* for NVIDIA developers contributing to
DOCA PCC itself, nor is it the place to learn how to *write*
the DPA-side congestion-control algorithm itself (that path
goes through the public DOCA PCC programming guide and the
companion DOCA DPA / DPACC guides via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

**Language scope.** DOCA PCC ships as a host-side C library
with `pkg-config` module name `doca-pcc`. The host-side API is
C; the DPA-side congestion-control algorithm is a separate
translation unit written in the language the DPACC compiler
accepts and compiled by `dpacc` into a binary that the host
packages into the executable as the PCC algorithm image. The
shipped samples under `/opt/mellanox/doca/samples/doca_pcc/`
are written in C plus DPA-side source (NVIDIA's choice).
Other-language consumers are limited in practice — the
DPA-side algorithm has no FFI escape hatch because it must be
a translation unit `dpacc` accepts — but a Rust / Go / Python
host-side wrapper that drives `doca_pcc_*` setup and loads a
PCC algorithm image built separately is still useful, and the
skill keeps the lifecycle, capability-discovery,
env-precondition, and error-taxonomy guidance language-neutral.

## When to load this skill

Load this skill when the user is doing hands-on DOCA PCC work
**from the host side** for a **custom** congestion control
algorithm, in any host language plus a DPA-side translation
unit built by `dpacc`. Concretely:

- Initializing a `doca_pcc` against a `doca_dev` that maps to
  the BlueField port carrying the RDMA / RoCE traffic to be
  controlled.
- Loading a PCC algorithm image (`doca_pcc_app`) that `dpacc`
  produced from the user's DPA-side congestion-control algorithm
  source, into the `doca_pcc` context.
- Parameterizing the loaded algorithm with the host-side knobs
  the algorithm exposes, and starting the `doca_pcc` Core
  context so the algorithm begins affecting RDMA / RoCE
  traffic on the attached port.
- Checking which PCC features are supported on the active
  `doca_devinfo` via the `doca_pcc_cap_*` family — BlueField
  generations and firmware revisions differ in whether they
  permit a custom PCC algorithm.
- Debugging a `DOCA_ERROR_*` returned from a `doca_pcc_*` call
  — in particular disambiguating *firmware-level custom-PCC
  slot disabled* from *device does not support custom PCC at
  all* from *algorithm image incompatible with this device*
  from *standard `doca_dev` access denied*.
- Designing host-side bindings in a non-C language that drive
  a custom PCC algorithm image they built separately with
  `dpacc` — the env-precondition and capability-discovery rules
  in this skill still apply.

Do **not** load this skill for general DOCA orientation, install
of DOCA or the DPACC compiler, the DPA-side programming model
itself (how to write the congestion-control algorithm body that
runs on the DPA), questions about the default factory PCC
algorithm shipped in ConnectX firmware (no host-side `doca-pcc`
code needed — that path is firmware-only configuration), or
questions about the `pcc_counters` diagnostic CLI (route via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)).
For all of those, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the matching upstream guide.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
PCC-specific material lives in two companion files:

- `CAPABILITIES.md` — what the host-side PCC API can express on
  this version + this BlueField generation + this firmware: the
  per-PCC-instance `doca_pcc` context, the loaded `doca_pcc_app`
  algorithm image produced by `dpacc`, the attach-to-port
  semantics that bind the algorithm to the RDMA / RoCE traffic
  it will control, the capability-query surface
  (`doca_pcc_cap_*`), the PCC error taxonomy mapped onto the
  cross-library `DOCA_ERROR_*` set, the observability surface
  (host-side reports plus the public PCC counter tool reachable
  via `doca-public-knowledge-map`), and the safety policy that
  gates env preconditions (DPA-capable BlueField, firmware-level
  custom-PCC slot enabled, matched DOCA + DPACC versions,
  algorithm image and host-side expectations agree).
- `TASKS.md` — step-by-step workflows for the six in-scope PCC
  verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host where DOCA is already installed at
the standard location, a BlueField with a DPA processor is
physically present and visible to the host, the BlueField
firmware has the custom-PCC slot enabled, the DPACC compiler
is installed at a version matched to the DOCA install per the
DOCA Compatibility Policy, and the user already knows how
(at least at a sketch level) to write the DPA-side PCC
algorithm that `dpacc` will compile. It does not cover
installing DOCA, installing the DPACC compiler, or flipping
firmware-level configuration — those paths go through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA PCC application source code or DPA-side
  algorithm source, in any language.** The verified PCC source
  is the shipped C + DPA-side samples at
  `/opt/mellanox/doca/samples/doca_pcc/`. The agent's job is to
  route the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the PCC-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **A specific congestion control algorithm.** This library
  *loads* an algorithm the user supplies; it does *not* define
  one. The agent must refuse to invent algorithm bodies and
  must route any *"what algorithm should I write"* question to
  the public DOCA PCC programming guide and the user's own
  domain expertise — that is a research question, not an API
  question.
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, …) parked inside the skill. The agent
  constructs the build manifest *in the user's project
  directory* against the user's installed DOCA + DPACC
  compiler, where `pkg-config --modversion doca-pcc` and the
  installed `dpacc` are the two sources of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will read
  it as buildable.
- **`pcc_counters` tool surface.** That CLI is a *separate
  artifact* (the real tool is the `pcc_counters.sh` script under
  `tools/pcc_counters/`) with its own public page; routing for it lives in
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
  Conflating it with the `doca-pcc` library is the single most
  common PCC first-app design error.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (host-side custom PCC work, not DPA-side algorithm
   design and not the counter tool).
2. **For the PCC capability matrix, the two-side-program model,
   the `doca_pcc` per-instance context, the loaded `doca_pcc_app`
   algorithm image, the attach-to-port semantics, the
   triple-axis precondition rule, the env-precondition policy,
   the error taxonomy, the observability surface, and the
   safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules (with the PCC overlay that DOCA
must match the DPACC compiler), and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
PCC programming guide, the public DPA / DPACC guides, the
`pcc_counters` tool guide, or in the on-disk install
layout" rather than "PCC host-side-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation
  source and the on-disk layout of an installed DOCA package.
  The PCC public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-PCC/index.html>; the
  `pcc_counters` diagnostic CLI lives under the DOCA Tools
  umbrella as a *companion surface* rather than a redefined
  artifact here.
- [`doca-dpa`](../doca-dpa/SKILL.md) — the host-side DPA
  control library that PCC depends on conceptually: the
  custom PCC algorithm *is* DPA-side code, compiled by the
  DPACC compiler, and the host-side `doca-pcc` Core lifecycle
  follows the same shape as the `doca-dpa` Core lifecycle. The
  agent loads `doca-dpa` alongside this skill when the user
  has DPA-level questions (kernel-launch model, DPA-side
  libraries) that PCC is layered on top of.
- [`doca-rdma`](../doca-rdma/SKILL.md) — the library whose
  traffic the custom PCC algorithm is controlling. The custom
  PCC algorithm affects RDMA / RoCE flows on the attached
  BlueField port; if the user has not yet set up RDMA / RoCE
  traffic on that port, there is nothing for the algorithm to
  act on, and `doca-rdma` is the skill that brings that
  traffic up.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, DPACC compiler install / verification,
  BlueField firmware configuration (including the custom-PCC
  slot enable), and the *I have no install yet* path with the
  public NGC DOCA container. This skill assumes its
  preconditions are satisfied AND that DPACC is installed at a
  version that matches DOCA AND that the firmware-level
  custom-PCC slot is enabled.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the PCC-specific *DOCA-and-DPACC must match* overlay per the
  DOCA Compatibility Policy (inherited from
  [`doca-dpa`](../doca-dpa/SKILL.md)).
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
  This skill layers PCC specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). PCC-specific debug (custom-PCC slot not
  enabled in firmware, DPACC + DOCA version skew, algorithm
  image rejected as incompatible with the device, traffic on
  the attached port not being affected because the algorithm
  body has no effect path) overlays on top of that ladder.

The default factory PCC algorithms shipped inside ConnectX
firmware are **not in scope** for this skill — those work
without `doca-pcc` and are configured through firmware-level
knobs, not through any host-side library API. The
`pcc_counters` diagnostic CLI is **also not in scope** —
it is a separate artifact for *inspecting* runtime PCC
counters and lives under the public DOCA Tools umbrella.
Conflating either of those with the `doca-pcc` library is the
single most common PCC first-app design error.

