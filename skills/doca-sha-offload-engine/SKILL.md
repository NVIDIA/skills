---
license: Apache-2.0
name: doca-sha-offload-engine
description: >
  Use this skill when wiring the DOCA SHA Offload Engine
  (an OpenSSL ENGINE) into an existing OpenSSL pipeline
  to offload one-shot SHA-1, SHA-256, or SHA-512
  (EVP_Digest) onto DOCA SHA hardware without rewriting
  against doca-sha. Covers engine load mechanics
  (`openssl engine dynamic`, `set_pci_addr` ctrl,
  `-engine_impl`), the SHA-224 negative test that proves
  offload engaged, the message-size window where offload
  beats CPU SHA, and engine-vs-library selection.
  Trigger even when the user does not say "DOCA SHA
  Offload Engine" or "OpenSSL ENGINE" — typical implicit
  phrasings: "speed up openssl SHA on BlueField",
  "offload SHA without code changes", "is openssl using
  the accelerator or falling back to software", "prove
  DOCA SHA actually ran", "openssl dgst hashed but I'm
  not sure it was offloaded". Refuse and route elsewhere
  for new SHA pipelines (use doca-sha), MD5 / SHA-3 /
  SHA-224 / HMAC-SHA offload, incremental hashing via
  chained `EVP_DigestUpdate`, or OpenSSL PROVIDER authoring.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on
  Linux (Ubuntu 22.04/24.04 or RHEL/SLES) with a
  BlueField DPU or ConnectX NIC attached, plus OpenSSL
  ≥ 1.1.1 and libssl-dev (or distro equivalent) on the
  build host. The engine ships under
  /opt/mellanox/doca/tools/doca_sha_offload_engine/ as
  libdoca_sha_offload_engine.so; reads the user's local
  install via `pkg-config doca-sha` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA SHA Offload Engine

**Where to start:** This is a tool skill for the OpenSSL
ENGINE shipped in the DOCA SOURCE tree under
`doca/tools/sha_offload_engine/` and INSTALLED on the host
under `${DOCA_DIR}/tools/doca_sha_offload_engine/` as
`libdoca_sha_offload_engine.so`. The directory-name shift
(`sha_offload_engine` in the source layout vs
`doca_sha_offload_engine` in the install layout) is an
NVIDIA packaging convention, not a bundle inconsistency;
both forms appear below and are the same artifact at
different lifecycle stages — quote whichever the prompt is
about (build-from-source vs runtime-load). It is **not a CLI** —
it is a shared object loaded by an OpenSSL-based
application or by `openssl` itself, that re-routes SHA-1 /
SHA-256 / SHA-512 (one-shot only, via the `EVP_Digest`
interface) onto the DOCA SHA hardware path. Open
[`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the PCIe-address
configuration and the OpenSSL prerequisites; jump to
[`## run`](TASKS.md#run) for the *"load the engine and
prove it actually runs"* flow. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what the engine actually offloads vs falls back to*,
*when the engine is a perf win vs not*, or *how to verify
offload actually engaged*. If DOCA is not installed yet,
route to [`doca-setup`](../../doca-setup/SKILL.md) first.
If the user is building a new SHA pipeline from scratch
(not wrapping an existing OpenSSL-based one), this skill
is the wrong surface — route to
[`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md)
instead.

## Example questions this skill answers well

The CLASSES of `doca-sha-offload-engine` questions this
skill is built to answer, each with one worked example.
The class is the load-bearing piece; the worked example
is one instance.

- **"I have an existing OpenSSL-based pipeline doing SHA;
  can I offload the SHA to DOCA without rewriting the
  app?"** — worked example: *"the app uses
  `EVP_DigestInit_ex` / `EVP_DigestUpdate` /
  `EVP_DigestFinal_ex` against `EVP_sha256()`; can I
  drop in DOCA-SHA offload via an engine load?"*.
  Answered by the *"when this engine is the right
  surface"* rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the engine-load mechanics in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Did the engine actually load? Or did OpenSSL just
  fall back to software SHA?"** — worked example: *"my
  `openssl dgst` invocation completed; how do I know
  DOCA SHA actually did the work and OpenSSL did not
  silently use the software path?"*. Answered by the
  *"prove the engine actually ran"* pattern in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  (the SHA-224 negative test and the `-engine_impl`
  flag) + the verification flow in
  [`TASKS.md ## test`](TASKS.md#test).
- **"For what message-size range is the engine offload a
  win vs CPU SHA?"** — worked example: *"my pipeline
  hashes 4 KB blocks at a time; is the engine a win
  there, or does the round-trip to DOCA SHA cost more
  than the CPU hash itself?"*. Answered by the
  message-size-window rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `openssl speed` perf-comparison pattern in
  [`TASKS.md ## test`](TASKS.md#test).
- **"What SHA algorithms does the engine actually
  support — and what happens for the ones it does not?"**
  — worked example: *"my pipeline mixes SHA-1, SHA-256,
  SHA-512, and SHA-224 — what does the engine do for
  each?"*. Answered by the algorithm-coverage matrix in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"Should I use this engine, or call doca-sha
  directly?"** — worked example: *"I am writing a new
  service from scratch; does the engine save me work or
  does it add complexity I do not need?"*. Answered by
  the *"engine vs library"* selection table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"What OpenSSL versions does the engine require, and
  what about OpenSSL 3.x's deprecation of the ENGINE
  API?"** — worked example: *"my host has OpenSSL 3.0;
  will the engine still load?"*. Answered by the version
  overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  (verified surface: the engine is documented for
  OpenSSL 1.1.1f on Ubuntu 20.04 and OpenSSL 3.0.2 on
  Ubuntu 22.04 per the shipped `readme.md`).

## Audience

This skill serves **external developers and operators who
have an existing OpenSSL-based pipeline doing SHA hashing
and want to offload SHA onto DOCA SHA without rewriting
their application against the doca-sha C API**.
Concretely:

- A developer integrating DOCA SHA acceleration into a
  service that already uses `EVP_Digest` for SHA-1 /
  SHA-256 / SHA-512.
- An operator deploying an `openssl dgst` / `openssl
  speed` based pipeline and wanting to measure the win
  from DOCA SHA offload without changing the pipeline's
  invocation surface.
- An SRE / performance engineer producing a *"this is
  the engine-offload win vs CPU SHA on this message-size
  mix"* artifact to inform a code-change decision (e.g.
  *"should we adopt the engine as-is, or rewrite to
  doca-sha for finer control?"*).
- An AI agent answering *"can I drop DOCA SHA into this
  OpenSSL-based app without code changes"* honestly —
  with the verified algorithm-coverage matrix, the
  message-size window, and the verification pattern that
  proves the engine actually ran.

It is **not** for users building a new SHA pipeline from
scratch (route to
[`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md)),
**not** for users wanting MD5 / SHA-2-224 / SHA-3 /
HMAC-SHA offload (the engine does not implement those —
the verified surface per the engine's source is one-shot
SHA-1, SHA-256, SHA-512 via `EVP_Digest`), and **not** a
substitute for the public DOCA SHA programming guide.

## Language scope

The DOCA SHA Offload Engine is shipped as a **C dynamic
shared object** (`libdoca_sha_offload_engine.so`) built
from `doca/tools/sha_offload_engine/{engine/doca_sha_offload_engine.c,
lib/doca_sha_offload_lib.{c,h}}` via `meson`. Its
*consumer* interface is OpenSSL's ENGINE API; any
language that calls OpenSSL (C, C++, Rust via `openssl`
crate, Python via `cryptography` and `pyca/cryptography`'s
backend, Node via `node:crypto`) can therefore *use* the
engine, provided the language binding either calls
`ENGINE_load_dynamic` / `ENGINE_by_id` directly or honors
an OpenSSL `engines` config that loads it. The skill's
language-neutral contribution is the engine-load mechanics
and the verification pattern; the OpenSSL ENGINE API is
the contract.

## When to load this skill

Load this skill when the user is — or the agent needs to
— deploy the DOCA SHA Offload Engine into an OpenSSL-based
pipeline on a host with DOCA installed and a SHA-capable
device. Concretely:

- Wiring the engine into an existing OpenSSL-based
  application or `openssl` CLI invocation, with the
  intent of no source-level code changes (or only the
  minimum `ENGINE_load_dynamic` / `ENGINE_by_id` block
  shown in the verified `readme.md`).
- Verifying the engine actually engaged for the
  hot-path digests (the *"is offload real or did OpenSSL
  fall back to software"* question).
- Characterizing the message-size range over which the
  engine is a perf win vs the CPU `sha1` / `sha256` /
  `sha512` paths.
- Deciding between *"adopt the engine"* (existing
  pipeline, minimal change) and *"rewrite to doca-sha"*
  (new pipeline, fine-grained control needed).

Do **not** load this skill for users building a new
SHA-pipeline from scratch — route to
[`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md).
Do not load this skill for users wanting algorithms the
engine does not implement (MD5, SHA-3, SHA-224, HMAC-SHA,
streaming/incremental SHA via `EVP_DigestUpdate` chains
that the engine treats as one-shot only).

## What this skill provides

This is a **thin loader**. Substantive material lives in
two companion files:

- `CAPABILITIES.md` — what the engine offloads (verified:
  one-shot SHA-1, SHA-256, SHA-512 via the OpenSSL
  `EVP_Digest` high-level interface), what it does NOT
  offload (anything else — including SHA-224, MD5,
  SHA-3, HMAC-SHA, and any chained
  `EVP_DigestInit_ex` / `EVP_DigestUpdate` /
  `EVP_DigestFinal_ex` pattern that the engine implements
  by buffering and then calling DOCA SHA in one shot at
  `Final`), the engine-vs-library selection rule, the
  message-size-window rule for when offload is a perf
  win, the verified ctrl-cmd surface (`set_pci_addr`),
  the version overlay (OpenSSL ≥ 1.1.1; the engine's
  shipped tests cover OpenSSL 1.1.1f and OpenSSL 3.0.2
  per the `readme.md`; OpenSSL 3.x deprecates the ENGINE
  API but still supports it via the legacy code path),
  the layered error taxonomy, the observability surface
  (the *"prove offload engaged"* pattern using the
  SHA-224 negative test and `-engine_impl`), and the
  safety overlay.
- `TASKS.md` — step-by-step workflows for the in-scope
  task verbs: `install`, `configure` (PCIe address
  selection — the engine defaults to `03:00.0` and
  exposes the `set_pci_addr` ctrl-cmd plus a build-time
  override per the shipped `test_cmdline_mode/readme.md`),
  `build` (the `meson` flow), `modify` (refuses source
  patching; modify the load-time invocation and the
  PCIe address instead), `run` (load the engine via
  `openssl engine dynamic`; the verification pattern;
  the OpenSSL programmatic `ENGINE_load_dynamic` block),
  `test` (the *"prove the engine ran"* SHA-224 negative
  test; the `openssl speed` perf comparison; the
  message-size window characterization), `debug` (walk
  the error taxonomy layer by layer), `use` (the
  engine-vs-library decision for the user's specific
  pipeline), plus a `Deferred task verbs` block.

The skill assumes a host where DOCA is already installed,
OpenSSL ≥ 1.1.1 is present (`libssl-dev` or equivalent),
and the operator has whatever privileges the public
install profile expects for the engine's PCIe binding.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or
scripts bundle. It deliberately does not contain — and
pull requests should not add:

- **Pre-written OpenSSL ENGINE application source code**
  beyond the verbatim verified block in the shipped
  `readme.md` (the `ENGINE_load_dynamic` /
  `ENGINE_by_id` / `ENGINE_ctrl_cmd_string` /
  `ENGINE_init` / `ENGINE_set_default_digests` sequence,
  cross-referenced into [`TASKS.md ## run`](TASKS.md#run)).
  The shipped readme is the worked example.
- **A `samples/`, `bindings/`, or `reference/`
  subtree.** This is a thin loader for a shipped
  shared-object; substantive material lives in the
  shipped `readme.md` and the doca-sha library docs.
- **Performance numbers from agent memory.** The
  message-size-window where the engine wins is
  device-, firmware-, OpenSSL-version-, and
  workload-specific. The `openssl speed` comparison
  pattern in [`TASKS.md ## test`](TASKS.md#test) is the
  way to capture it on the user's actual hardware;
  quoting a number from memory is the cross-platform
  failure mode this skill exists to prevent.
- **Wrappers, parsers, or scripts** in any language
  that consume the engine's stdout or the `openssl
  speed` output format.

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (the user actually has an
   existing OpenSSL-based pipeline and wants to offload
   to DOCA SHA *without* code changes; if the user is
   building from scratch, route to
   [`../../libs/doca-sha/`](../../libs/doca-sha/SKILL.md)).
2. **For what the engine offloads vs falls back, the
   engine-vs-library selection rule, the message-size-
   window rule, the version overlay, the error taxonomy,
   the observability surface (and the prove-offload-
   actually-engaged pattern), and the safety overlay,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `install`,
   `configure`, `build`, `modify`, `run`, `test`,
   `debug`, `use` — see [TASKS.md](TASKS.md).**

## Related skills

- [`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md) —
  the underlying DOCA SHA library. The engine is a
  thin OpenSSL-ENGINE wrapper around doca-sha; when the
  user needs fine-grained control over the SHA task
  surface (partial-hash, custom buffer permissions,
  cap-query for unusual message sizes), the library is
  the right answer. The engine wraps the *one-shot*
  task; the library exposes both one-shot and partial-
  hash per
  [`../../libs/doca-sha/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-sha/CAPABILITIES.md#capabilities-and-modes).
- [`doca-version`](../../doca-version/SKILL.md) — the
  canonical version-detection chain. The engine has a
  TWO-axis version overlay (DOCA-side and OpenSSL-side);
  the version skill carries the four-way match rule
  this skill layers on top of.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. The engine surfaces its
  own error taxonomy; when the cause is below DOCA
  (driver, firmware), the taxonomy hands off here.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, the `libssl-dev`
  install path, and the NGC DOCA container path.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing to the public DOCA SHA documentation set on
  `docs.nvidia.com/doca/sdk/` and to the OpenSSL
  ENGINE / `openssl-engine` upstream documentation on
  `openssl.org`.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the bundle-wide hardware-safety meta-policy. The
  engine binds to a specific PCIe device; the
  `set_pci_addr` ctrl is the artifact-specific overlay,
  but any host-side change underneath (firmware burn,
  BFB reflash) runs through the meta-policy.
