---
license: Apache-2.0
name: doca-rmax
description: >
  Use this skill when the user is doing hands-on DOCA Rivermax work on
  a BlueField DPU or ConnectX host — standing up `doca_rmax_in_stream`
  (receive) sessions for timing-precise
  media-over-IP (SMPTE ST 2110 video/audio, market data, scientific
  feeds), confirming the Rivermax SDK + license precondition before
  any DOCA-side code, running `doca_rmax_get_*_supported` capability queries,
  pairing with `doca-eth` queues and `doca-flow` steering, or
  debugging `DOCA_ERROR_*` from a Rivermax call. Trigger even when the
  user does not explicitly mention "DOCA Rivermax" or "rmax" —
  implicit phrasings include "ST 2110 receive isn't getting frames",
  "sub-microsecond jitter on BlueField", "NOT_PERMITTED on first
  rivermax create", "no recv events after stream start", or "license
  check failing on a media receiver". Refuse and route elsewhere for
  installing the Rivermax SDK or its license, programming the
  underlying queue (`doca-eth`), steering rules (`doca-flow`), or
  best-effort packet I/O — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK at /opt/mellanox/doca on Linux (Ubuntu 22.04/24.04 or
  RHEL/SLES) with a BlueField DPU or ConnectX NIC, AND the separately-
  installed NVIDIA Rivermax SDK with a valid Rivermax license readable by
  the user — DOCA does NOT bundle Rivermax. Reads the local install via
  `pkg-config doca-rmax`; route Rivermax SDK install/license questions to
  the public Rivermax guide.

---

# DOCA Rivermax

**Where to start:** This skill assumes DOCA is already installed,
**AND** that the NVIDIA Rivermax SDK is separately installed with a
valid Rivermax license present on the host. The agent's FIRST
action on any Rivermax question is to confirm both — without
Rivermax SDK + license, `doca-rmax` cannot function regardless
of how clean the DOCA-side code is, and this is the #1 first-app
confusion (DOCA does *not* bundle Rivermax; it wraps it). Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something
(configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
can DOCA Rivermax express* on this version + this Rivermax
install. If the user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; for the Rivermax
SDK + license install itself, route to the public DOCA Rivermax
guide (slug `DOCA-Rivermax`) via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
If the user is asking *"how do I get packets to land on my
Rivermax input stream at all"*, the answer is layered:
`doca-rmax` is the *Rivermax integration* surface,
[`doca-eth`](../doca-eth/SKILL.md) is the *queue* surface that
carries the packets, and [`doca-flow`](../doca-flow/SKILL.md)
is the *steering* surface that directs them.

## Example questions this skill answers well

The CLASSES of DOCA Rivermax questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a
single instance.

- **"Can I even use `doca-rmax` on this host?"** — worked
  example: *"the public docs mention `doca-rmax`; is it
  usable without doing anything else?"*. Answered by the
  Rivermax-SDK + license precondition rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes the install-side question to the public Rivermax guide
  via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  and refuses to recommend a fallback to `doca-eth` alone that
  would silently lose the timing properties.
- **"How do I set up a SMPTE ST 2110 video receive stream?"** —
  worked example: *"line up an inbound Rivermax stream on a
  representor of a BlueField port for first-run testing"*.
  Answered by the per-stream object lifecycle in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  input stream capability table.
- **"Which `doca_rmax_get_*_supported` query do I have to call before
  picking a stream type / frame size / packet rate?"** — worked
  example: *"can this device + this Rivermax install carry a
  1080p60 video stream at the rate I want?"*. Answered by the
  capability-query rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + step 3 in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"My stream is set up but no packets arrive — why?"** —
  worked example: *"the Rivermax stream object started cleanly
  but no recv events fire"*. Answered by the precondition matrix
  in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes the steering side to
  [`doca-flow`](../doca-flow/SKILL.md), the queue side to
  [`doca-eth`](../doca-eth/SKILL.md), and the license side
  back to the Rivermax-side precondition.
- **"Is this Rivermax integration capability available on my
  device + my installed DOCA + my Rivermax SDK?"** — worked
  example: *"does this device + this Rivermax version advertise
  the stream type I need"*. Answered by the capability-query
  rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the version-and-device overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which adds the *Rivermax-side version is a second axis* rule
  on top of the canonical DOCA version-handling chain in
  [`doca-version`](../../doca-version/SKILL.md).
- **"What does this `DOCA_ERROR_*` from a Rivermax call mean and
  which layer caused it?"** — worked example:
  *"`DOCA_ERROR_NOT_PERMITTED` on the first Rivermax stream
  create"*. Answered by the Rivermax overlay on the
  cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md), and which calls out
  that `_NOT_PERMITTED` on Rivermax frequently means *"Rivermax
  license missing / expired / not readable"*, not the usual
  DOCA-side device-access shortfall.

## Audience

This skill serves **external developers building applications
that consume the DOCA Rivermax integration** — i.e., users whose
code calls `doca_rmax_*` (directly in C/C++, or through
FFI/bindings from another language) to drive timing-precise
media-over-IP streams (SMPTE ST 2110, real-time market data,
high-throughput scientific instrument streams) on top of a
separately-installed NVIDIA Rivermax SDK on a BlueField or
ConnectX host. It is *not* for NVIDIA developers contributing to
DOCA Rivermax itself, and it is *not* for users who want
best-effort packet I/O — for that, route to
[`doca-eth`](../doca-eth/SKILL.md) directly.

**Language scope.** DOCA Rivermax ships as a C library with
`pkg-config` module name `doca-rmax`. The shipped samples
live under `/opt/mellanox/doca/samples/doca_rmax/` and are
written in C. C and C++ consumers are the canonical case; the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the precondition rule (Rivermax SDK +
license), per-stream lifecycle, capability-discovery,
permission, scheduling-discipline, and error-taxonomy guidance
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Rivermax
work, in any language. Concretely:

- Confirming that the NVIDIA Rivermax SDK and a valid Rivermax
  license are present on the host before any DOCA-side code is
  written — this is a **mandatory precondition**, not an error
  path; without it the answer is *"`doca-rmax` cannot be
  used; pick a different library"*, not *"let's try and see what
  fails"*.
- Initializing the global DOCA Rivermax engine with
  `doca_rmax_init()` / `doca_rmax_release()`, then creating a
  `doca_rmax_in_stream` (receive) context with
  `doca_rmax_in_stream_create()` on a `doca_dev` opened against a
  physical port, a representor, or an SF, and converting it via
  `doca_rmax_in_stream_as_ctx()` before `doca_ctx_start()`. The
  public DOCA Rivermax API is receive-only — there is no
  transmit/output stream object.
- Configuring a Rivermax input stream for SMPTE ST 2110 audio +
  video over IP, real-time market data feeds, or a scientific
  instrument stream — with awareness that the stream type,
  frame size, and packet rate the user wants are *all* device-
  and Rivermax-conditional and gate on the matching
  `doca_rmax_get_*_supported` query.
- Reading or setting Rivermax stream properties via
  `doca_rmax_in_stream_set_*` and querying device + Rivermax
  capability via the `doca_rmax_get_*_supported` family before
  assuming any specific stream type / frame size / packet rate
  is supported.
- Pairing the Rivermax stream with the `doca-eth` queue surface
  that carries the packets and the `doca-flow` rules that
  steer them — Rivermax does not program steering itself.
- Designing the real-time scheduling discipline for the
  streaming threads (the underlying Rivermax stack expects
  real-time priority for sub-microsecond jitter; the canonical
  scheduling guidance lives in the Rivermax SDK docs reachable
  through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
- Debugging a `DOCA_ERROR_*` returned from a Rivermax call
  (lifecycle vs. license / permission vs. capability vs.
  driver-below) where the cause may live in the DOCA-side
  wrapper, the underlying Rivermax stack, or the licensing
  layer.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the DOCA Rivermax C ABI — for the precondition
  rule, lifecycle, capability, permission, and scheduling rules
  the wrapper must honor.

Do **not** load this skill for general DOCA orientation, for
installing DOCA itself, for installing the Rivermax SDK or
managing the Rivermax license file (those live in the public
Rivermax SDK guide reachable through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)),
for best-effort packet I/O without timing requirements (use
[`doca-eth`](../doca-eth/SKILL.md) directly), or for pure
host-side data processing without networking. For DOCA
documentation orientation, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
Rivermax-specific material lives in two companion files:

- `CAPABILITIES.md` — what DOCA Rivermax can express on this
  version *and this Rivermax SDK install*: the
  Rivermax-as-hard-dependency rule, the receive-only
  `doca_rmax_in_stream` object model, the capability-query surface
  (`doca_rmax_get_*_supported`), the Rivermax error taxonomy
  (mapped onto the cross-library `DOCA_ERROR_*` set, with
  Rivermax-specific causes called out per row), the
  observability surface (per-stream progress engine events,
  capability snapshots, Rivermax-side license + driver state),
  and the safety policy that gates the Rivermax-SDK-present /
  license-present / device-access / scheduling-discipline
  preconditions.
- `TASKS.md` — step-by-step workflows for the six in-scope
  Rivermax verbs: `configure`, `build`, `modify`, `run`,
  `test`, `debug`. Plus a `Deferred task verbs` block that
  points out-of-scope questions (installing Rivermax,
  managing the license, programming steering, programming the
  underlying queue) at the right next skill, and a `Command
  appendix` of the recurring commands the agent reaches for.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location, the NVIDIA Rivermax SDK is
installed at its expected location with a valid license, and
the user has the privileges their public install profile
expects (typically sudo or `mlnx`-group membership to open a
`doca_dev` against a port). It does not cover installing DOCA —
that path goes through
[`doca-setup`](../../doca-setup/SKILL.md). It does not cover
installing Rivermax or its license — that path goes through the
public Rivermax SDK guide reachable through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Rivermax application source code, in any
  language.** The verified Rivermax source code is the shipped
  C samples at `/opt/mellanox/doca/samples/doca_rmax/<name>/`.
  The agent's job is to route the user to those files and
  prescribe a minimum-diff modification on them via the
  universal modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the Rivermax-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-rmax` is the source of
  truth.
- **Quoted Rivermax symbol names beyond the public family
  pattern (`doca_rmax_get_*_supported` for capability discovery,
  receive-only `doca_rmax_in_stream` session objects driven by the
  standard DOCA Core context lifecycle).** Exact Rivermax
  symbol names are install-bound and Rivermax-SDK-version-
  bound; the agent should read them from the installed headers
  at $(pkg-config --variable=includedir doca-common) and from the
  public DOCA Rivermax guide rather than rely on agent memory.
- **A `samples/`, `bindings/`, or `reference/` subtree** of
  any kind. A mock or incomplete artifact in this skill's
  tree, even one labeled *"reference"*, is misleading: users
  will read it as buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope **and** that the Rivermax-SDK + license
   precondition has been considered.
2. **For the Rivermax capability matrix, the receive-only input
   stream model, capability-query rules, error taxonomy,
   observability, and safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules (Rivermax adds a Rivermax-SDK-
version axis on top),
[`doca-eth`](../doca-eth/SKILL.md) for the queue surface
that carries the packets,
[`doca-flow`](../doca-flow/SKILL.md) for the steering side
that decides which packets land on which queue, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is *"look it up in the public
Rivermax SDK guide or the installed package layout"* rather
than *"Rivermax-integration-specific guidance"*.

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  Rivermax URL slug is `DOCA-Rivermax`; the public Rivermax SDK
  guide (separate product) is reachable from the same routing
  table when the user asks how to install Rivermax or its
  license.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, port-state checks (`devlink dev show`,
  `ip link`), permission and group-membership requirements for
  opening a `doca_dev`. This skill assumes its preconditions
  are satisfied; the Rivermax-SDK + license preconditions are
  layered on top.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the Rivermax-specific overlay (Rivermax SDK version is a
  second axis; the capability set on this host is the
  intersection of DOCA-side cap-query results and
  Rivermax-side capabilities).
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill
  layers Rivermax specifics on top.
- [`doca-eth`](../doca-eth/SKILL.md) — the queue surface
  that carries the packets a Rivermax stream produces or
  consumes. DOCA Rivermax does *not* program the underlying
  queue itself; it integrates with the queue programmed via
  `doca-eth`. The two skills' lifecycles are independent.
- [`doca-flow`](../doca-flow/SKILL.md) — the steering
  surface that decides which packets land on which queue.
  DOCA Rivermax does *not* program steering itself; an empty
  Rivermax input stream almost always means a missing or wrong
  Flow rule (or a missing Rivermax license), not a Rivermax
  bug.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder (install / version / build /
  link / runtime / program / driver). Rivermax-specific debug
  (license precondition gaps, stream-type / packet-rate
  capability mismatches, scheduling-discipline jitter
  symptoms) overlays on top of that ladder.

