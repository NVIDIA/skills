---
license: Apache-2.0
name: doca-comch
description: >
  Use this skill when the user is doing hands-on DOCA Comch work
  on a host + BlueField pair — bringing up host ↔ DPU PCIe
  control-plane messaging, picking server (DPU) vs client (host)
  roles, choosing slow-path send-task / recv-callback vs
  fast-path producer / consumer, querying max-msg-size or
  max-clients capabilities, registering connection callbacks, or
  debugging DOCA_ERROR_* returns from the Comch API. Trigger
  even when the user does not explicitly mention "DOCA Comch" or
  "Comm Channel" (renamed in DOCA 2.5) — typical implicit
  phrasings include "send a control message from host to
  BlueField over PCIe", "DPU can't see the host representor",
  "DOCA_ERROR_NOT_PERMITTED on server_create", "DOCA_ERROR_AGAIN
  on task_send submit", "connect callback never fires", or
  "stream bulk data from a host driver to a DPU agent". Refuse
  and route elsewhere for installing DOCA itself, BFB / firmware
  bring-up, non-Comch DOCA libraries, or deploying Comch apps at
  scale — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) on a host + BlueField pair
  (the Comch channel runs over the RoCE/IB protocol, not the
  TCP/IP stack). Reads the user's
  local install via `pkg-config doca-comch` (legacy
  `doca-comm-channel` on installs <2.5) and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Comch

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on Comch work** on a BlueField + host
pair with DOCA. Open [`TASKS.md`](TASKS.md) if the user wants to
*do* something (configure / build / modify / run / test / debug);
open [`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what can Comch express* on this version. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"is Comch even on this DOCA"* because the docs they read
mention `doca-comm-channel`, route to
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
for the 2.5 rename.

## Example questions this skill answers well

The CLASSES of Comch questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"How do I bring up a Comch channel between host and DPU?"** —
  worked example: *"server side on the DPU, client side on the
  host, exchange a first control message"*. Answered by the
  role-selection + lifecycle workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  server-vs-client table.
- **"How do I move bulk data over Comch with low CPU?"** — worked
  example: *"stream a 64 KiB chunk every 100 µs from the host
  driver to a DPU agent"*. Answered by the producer / consumer
  fast-path described in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the channel-setup workflow in
  [`TASKS.md ## configure`](TASKS.md#configure)
  step 4, with the *"slow-path vs fast-path"* selection rule.
- **"What is the maximum message size I can send?"** — worked
  example: *"can I send a 4 MiB control message in one shot?"*.
  Answered by the capability-query rule
  (`doca_comch_cap_get_max_msg_size`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the property-set workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"Why doesn't the DPU side see the representor?"** — worked
  example: *"`doca_comch_server_create` returns
  `DOCA_ERROR_NOT_PERMITTED` on a freshly imaged BlueField"*.
  Answered by the representor + permission overlay in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes representor-side env questions to
  [`doca-setup`](../../doca-setup/SKILL.md).
- **"Is this Comch capability on my installed DOCA version?"** —
  worked example: *"is `doca_comch_producer` in DOCA 2.6.0, or do
  I still need the old slow-path API?"*. Answered by the
  version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  Comch-specific 2.5 rename rule.
- **"What does this `DOCA_ERROR_*` from a Comch call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_AGAIN`
  on submitting a `doca_comch_task_send` via `doca_task_submit`"*. Answered by the Comch
  overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA Comch library** — i.e., users whose code calls
`doca_comch_*` (directly in C/C++, or through FFI/bindings from
another language) to exchange control or data messages between a
host process and a BlueField agent over PCIe. It is *not* for
NVIDIA developers contributing to DOCA Comch itself.

**Language scope.** DOCA Comch ships as a C library with
`pkg-config` module name `doca-comch`. The shipped samples are
written in C. C and C++ consumers are the canonical case; the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the role-split, lifecycle,
capability-discovery, permission, and error-taxonomy guidance
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Comch work,
in any language. Concretely:

- Initializing a `doca_comch_server` on the DPU side or a
  `doca_comch_client` on the host side, on a representor or PCIe
  address the host can reach.
- Configuring at least one of: a recv callback for slow-path
  messages, a producer for fast-path outbound data, a consumer for
  fast-path inbound data, before `doca_ctx_start()`.
- Reading or setting comch properties via `doca_comch_set_*` and
  `doca_comch_cap_get_*` — max message size, max number of
  clients (server side), producer / consumer queue sizing.
- Establishing a connection between the two sides and reacting to
  connection-state transitions via the connection callbacks
  registered before start.
- Choosing between the **slow-path** (send-task / recv-callback,
  message-oriented, lower throughput, single-call simplicity) and
  the **fast-path** (producer / consumer, asynchronous, much
  higher throughput, two-context setup).
- Debugging a `DOCA_ERROR_*` returned from a Comch call (lifecycle
  vs. permission vs. capability vs. would-block) and the
  connection state machine across the host / DPU pair.
- Designing or extending non-C bindings (Rust, Go, Python, …) that
  wrap the Comch C ABI — for the lifecycle, role-split,
  permission, and capability rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, or non-Comch library questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive Comch-specific
material lives in two companion files:

- `CAPABILITIES.md` — what Comch can express on this version: the
  server vs client role split, the slow-path send-task /
  recv-callback surface, the producer / consumer fast-path
  surface, the capability-query surface
  (`doca_comch_cap_get_*`), the Comch error taxonomy (mapped onto
  the cross-library `DOCA_ERROR_*` set), the observability surface
  (connection state callbacks, task completion callbacks), and
  the safety policy that gates representor and permission
  decisions.
- `TASKS.md` — step-by-step workflows for the six in-scope Comch
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host + BlueField pair where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects (in particular, sudo on the
DPU side to see the host representor). It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Comch application source code, in any
  language.** The verified Comch source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_comch/<name>/`. The
  agent's job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the Comch-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against
  the user's installed DOCA, where `pkg-config --modversion
  doca-comch` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the Comch capability matrix, role split, slow-path /
   fast-path surfaces, permission policy, error taxonomy,
   observability, and safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "Comch-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package. The Comch URL
  slug is `DOCA-Comch` (DOCA 2.5+), not `doca-comm-channel`.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, representor visibility checks, and the *I
  have no install yet* path with the public NGC DOCA container.
  This skill assumes its preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds the
  Comch-specific 2.5 rename overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers Comch specifics on
  top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Comch-specific debug (lifecycle violations,
  representor visibility, slow-path vs fast-path queue-full
  symptoms) overlays on top of that ladder.
